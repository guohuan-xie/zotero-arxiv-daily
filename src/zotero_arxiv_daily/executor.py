from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, _request_llm
import random
import re
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    def score_paper_quality(self, paper) -> float | None:
        """Score methodological quality from 0 to 10 using a strict LLM review."""
        paper_text = (
            f"Title: {paper.title}\n\n"
            f"Abstract: {paper.abstract or ''}\n\n"
            f"Main-content preview: {(paper.full_text or '')[:6000]}"
        )
        prompt = (
            "Assess the overall scientific quality of this newly posted paper. "
            "Use a strict 0-10 scale and consider methodological rigor (30%), "
            "strength and completeness of empirical evidence (25%), novelty and "
            "potential significance (25%), and clarity/reproducibility (20%). "
            "Do not reward buzzwords, author identity, or institutional prestige. "
            "Penalize vague claims, weak comparisons, missing evidence, and "
            "unclear contributions. A score of 7.5 or above should be reserved "
            "for papers that appear genuinely strong from the available evidence. "
            "Return only one numeric score from 0 to 10.\n\n"
            + paper_text
        )
        try:
            response = _request_llm(
                self.openai_client,
                self.config.llm,
                [
                    {
                        "role": "system",
                        "content": "You are a strict and conservative scientific peer reviewer.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            match = re.search(r"\b(?:10(?:\.0+)?|[0-9](?:\.\d+)?)\b", response.strip())
            if match is None:
                raise ValueError(f"Cannot parse quality score from response: {response!r}")
            score = min(10.0, max(0.0, float(match.group(0))))
            logger.info(f"Quality score {score:.1f}/10 for {paper.title}")
            return score
        except Exception as e:
            logger.warning(f"Failed to assess quality of {paper.url}: {e}")
            return None

    def filter_high_quality_papers(self, papers):
        """Keep only papers that clear a configurable, conservative quality bar."""
        max_paper_num = int(self.config.executor.max_paper_num)
        if not self.config.executor.get("quality_filter", True):
            return papers[:max_paper_num]

        min_score = float(self.config.executor.get("min_quality_score", 7.5))
        candidate_num = int(
            self.config.executor.get(
                "quality_candidate_num",
                max(20, max_paper_num * 3),
            )
        )
        selected = []
        logger.info(
            f"Strict quality screening enabled: threshold={min_score:.1f}/10, "
            f"candidates={min(candidate_num, len(papers))}"
        )
        for paper in tqdm(papers[:candidate_num], desc="Quality screening"):
            score = self.score_paper_quality(paper)
            if score is not None and score >= min_score:
                paper.quality_score = score
                selected.append(paper)
                if len(selected) >= max_paper_num:
                    break
        logger.info(
            f"Selected {len(selected)} high-quality papers from "
            f"{min(candidate_num, len(papers))} candidates"
        )
        return selected

    
    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.warning(
                "No Zotero papers with abstracts were found; using a "
                "video-generation research profile as the recommendation seed."
            )
            corpus = [
                CorpusPaper(
                    title="Video generation research profile",
                    abstract=(
                        "Research on generative video models, including text-to-video and "
                        "image-to-video generation, diffusion transformers, controllable "
                        "generation, motion control, temporal consistency, world models, "
                        "video editing, personalization, stylized video generation, and "
                        "efficient training and inference."
                    ),
                    added_date=datetime.now(),
                    paths=["configured-interest/video-generation"],
                )
            ]
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = self.filter_high_quality_papers(reranked_papers)
            if len(reranked_papers) == 0 and not self.config.executor.send_empty:
                logger.info("No papers passed the quality threshold. No email will be sent.")
                return
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
