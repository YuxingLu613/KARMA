"""
KARMA Main Pipeline Implementation

This module contains the main KARMA pipeline that orchestrates all agents
to process documents and extract knowledge.
"""

import logging
import time
import os
from typing import List, Union, Optional, Dict
from pathlib import Path

from openai import OpenAI

from .data_structures import (
    KnowledgeTriple, KGEntity, IntermediateOutput,
    DocumentMetadata, ProcessingMetrics, KnowledgeGraph
)

from ..agents import (
    IngestionAgent, ReaderAgent, SummarizerAgent,
    EntityExtractionAgent, RelationshipExtractionAgent,
    SchemaAlignmentAgent, ConflictResolutionAgent, EvaluatorAgent
)

from karma.utils.pdf_reader import PDFReader

logger = logging.getLogger(__name__)


class KARMAPipeline:
    """
    Main KARMA pipeline for automated knowledge graph enrichment.

    This class orchestrates the entire pipeline:
    1. Ingestion -> 2. Reading -> 3. Summarization -> 4. Entity Extraction
    -> 5. Relationship Extraction -> 6. Schema Alignment
    -> 7. Conflict Resolution -> 8. Evaluation -> Integration
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o",
        integration_threshold: float = 0.6
    ):
        """
        Initialize KARMA pipeline with API credentials.

        Args:
            api_key: OpenAI API key
            base_url: Optional API base URL for Azure or custom endpoints
            model_name: Model identifier
            integration_threshold: Minimum score to integrate knowledge
        """
        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name
        self.integration_threshold = integration_threshold

        # Initialize all agent instances
        self._initialize_agents()

        # Initialize knowledge graph and utilities
        self.knowledge_graph = KnowledgeGraph()
        self.pdf_reader = PDFReader()

        # Track processing
        self.output_log: List[str] = []

    @classmethod
    def from_config(cls, config) -> 'KARMAPipeline':
        """
        Create pipeline from configuration object.

        Args:
            config: KARMAConfig instance

        Returns:
            Configured pipeline instance
        """
        # Validate and setup configuration
        config.validate()
        config.setup_logging()

        # Create pipeline instance
        pipeline = cls(
            api_key=config.model.api_key,
            base_url=config.model.base_url,
            model_name=config.model.name,
            integration_threshold=config.pipeline.integration_threshold
        )

        return pipeline

    def _initialize_agents(self):
        """Initialize all KARMA agents."""
        self.ingestion_agent = IngestionAgent(self.client, self.model_name)
        self.reader_agent = ReaderAgent(self.client, self.model_name)
        self.summarizer_agent = SummarizerAgent(self.client, self.model_name)
        self.entity_extraction_agent = EntityExtractionAgent(self.client, self.model_name)
        self.relationship_extraction_agent = RelationshipExtractionAgent(self.client, self.model_name)
        self.schema_alignment_agent = SchemaAlignmentAgent(self.client, self.model_name)
        self.conflict_resolution_agent = ConflictResolutionAgent(self.client, self.model_name)
        self.evaluator_agent = EvaluatorAgent(
            self.client, self.model_name,
            integrate_threshold=self.integration_threshold
        )

    def process_document(
        self,
        source: Union[str, Path],
        domain: str = "biomedical",
        relevance_threshold: float = 0.2
    ) -> IntermediateOutput:
        """
        Process a document and extract knowledge.

        Args:
            source: Text content or path to PDF/text file
            domain: Domain context for relevance scoring
            relevance_threshold: Minimum relevance score for segments

        Returns:
            IntermediateOutput containing all pipeline results
        """
        start_time = time.time()
        intermediate = IntermediateOutput()

        try:
            # Step 1: Load and ingest document
            raw_text = self._load_document(source)
            intermediate.raw_text = raw_text

            self._log("Starting KARMA pipeline...")

            # Step 2: Ingestion - Extract metadata and normalize text
            self._log("[1/8] Running Ingestion Agent...")
            step_start = time.time()

            metadata, normalized_content = self.ingestion_agent.process(raw_text)
            intermediate.metadata = metadata

            step_time = time.time() - step_start
            intermediate.metrics.add_agent_time("ingestion", step_time)
            self._log(f"[1/8] Ingestion completed in {step_time:.2f}s")

            # Step 3: Reader - Segment and score relevance
            self._log("[2/8] Running Reader Agent...")
            step_start = time.time()

            all_segments, relevant_segments = self.reader_agent.process(
                normalized_content, relevance_threshold
            )
            intermediate.segments = all_segments
            intermediate.relevant_segments = relevant_segments

            step_time = time.time() - step_start
            intermediate.metrics.add_agent_time("reader", step_time)
            self._log(f"[2/8] Reader completed in {step_time:.2f}s. "
                     f"Segments: {len(all_segments)}, Relevant: {len(relevant_segments)}")

            # Step 4: Summarizer - Create concise summaries
            self._log("[3/8] Running Summarizer Agent...")
            step_start = time.time()

            summaries = self.summarizer_agent.process(relevant_segments)
            intermediate.summaries = summaries

            step_time = time.time() - step_start
            intermediate.metrics.add_agent_time("summarizer", step_time)
            self._log(f"[3/8] Summarizer completed in {step_time:.2f}s. "
                     f"Summaries: {len(summaries)}")

            # Step 5: Entity Extraction - Identify entities
            self._log("[4/8] Running Entity Extraction Agent...")
            step_start = time.time()

            entities = self.entity_extraction_agent.process(summaries)
            intermediate.entities = entities

            step_time = time.time() - step_start
            intermediate.metrics.add_agent_time("entity_extraction", step_time)
            self._log(f"[4/8] Entity extraction completed in {step_time:.2f}s. "
                     f"Entities: {len(entities)}")

            # Step 6: Relationship Extraction - Find relationships
            self._log("[5/8] Running Relationship Extraction Agent...")
            step_start = time.time()

            relationships = self.relationship_extraction_agent.process(summaries, entities)
            intermediate.relationships = relationships

            step_time = time.time() - step_start
            intermediate.metrics.add_agent_time("relationship_extraction", step_time)
            self._log(f"[5/8] Relationship extraction completed in {step_time:.2f}s. "
                     f"Relationships: {len(relationships)}")

            # Step 7: Schema Alignment - Align to standard schema
            self._log("[6/8] Running Schema Alignment Agent...")
            step_start = time.time()

            aligned_entities, aligned_relationships = self.schema_alignment_agent.process(
                entities, relationships
            )
            intermediate.aligned_entities = aligned_entities
            intermediate.aligned_triples = aligned_relationships

            step_time = time.time() - step_start
            intermediate.metrics.add_agent_time("schema_alignment", step_time)
            self._log(f"[6/8] Schema alignment completed in {step_time:.2f}s")

            # Step 8: Conflict Resolution - Handle contradictions
            self._log("[7/8] Running Conflict Resolution Agent...")
            step_start = time.time()

            resolved_relationships = self.conflict_resolution_agent.process(
                aligned_relationships, self.knowledge_graph.triples
            )
            intermediate.final_triples = resolved_relationships

            step_time = time.time() - step_start
            intermediate.metrics.add_agent_time("conflict_resolution", step_time)
            self._log(f"[7/8] Conflict resolution completed in {step_time:.2f}s. "
                     f"Non-conflicting: {len(resolved_relationships)}")

            # Step 9: Evaluation - Final quality assessment
            self._log("[8/8] Running Evaluator Agent...")
            step_start = time.time()

            integrated_triples = self.evaluator_agent.process(resolved_relationships)
            intermediate.integrated_triples = integrated_triples

            step_time = time.time() - step_start
            intermediate.metrics.add_agent_time("evaluator", step_time)
            self._log(f"[8/8] Evaluation completed in {step_time:.2f}s. "
                     f"Integrated: {len(integrated_triples)}")

            # Step 10: Update Knowledge Graph
            self._update_knowledge_graph(aligned_entities, integrated_triples)

            # Finalize metrics
            total_time = time.time() - start_time
            intermediate.metrics.processing_time = total_time

            self._log(f"KARMA pipeline completed in {total_time:.2f}s. "
                     f"Added {len(integrated_triples)} knowledge triples.")

            return intermediate

        except Exception as e:
            intermediate.metrics.error_count += 1
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _load_document(self, source: Union[str, Path]) -> str:
        """
        Load document content from various sources.

        Args:
            source: File path or text content

        Returns:
            Raw text content
        """
        if isinstance(source, (str, Path)):
            source_path = Path(source)

            # Check if it's a file path
            if source_path.exists():
                if source_path.suffix.lower() == '.pdf':
                    return self.pdf_reader.extract_text(source_path)
                else:
                    # Assume text file
                    with open(source_path, 'r', encoding='utf-8') as f:
                        return f.read()
            else:
                # Treat as raw text if not a valid file path
                return str(source)
        else:
            return str(source)

    def _update_knowledge_graph(self, entities: List[KGEntity], triples: List[KnowledgeTriple]):
        """
        Update the internal knowledge graph with new knowledge.

        Args:
            entities: New entities to add
            triples: New triples to add
        """
        # Add entities
        for entity in entities:
            self.knowledge_graph.add_entity(entity)

        # Add triples
        for triple in triples:
            self.knowledge_graph.add_triple(triple)

    def _log(self, message: str):
        """
        Log message to both logger and internal log.

        Args:
            message: Message to log
        """
        logger.info(message)
        self.output_log.append(message)

    def get_knowledge_graph(self) -> KnowledgeGraph:
        """
        Get the current knowledge graph.

        Returns:
            Current knowledge graph
        """
        return self.knowledge_graph

    def clear_knowledge_graph(self):
        """Clear the knowledge graph."""
        self.knowledge_graph = KnowledgeGraph()
        self._log("Knowledge graph cleared")

    def export_knowledge_graph(self, output_path: Union[str, Path]) -> Dict:
        """
        Export knowledge graph to file.

        Args:
            output_path: Path to save the knowledge graph

        Returns:
            Dictionary representation of the knowledge graph
        """
        self.knowledge_graph.save_to_file(output_path)
        return self.knowledge_graph.to_dict()

    def get_pipeline_statistics(self) -> Dict:
        """
        Get statistics about the pipeline and knowledge graph.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'knowledge_graph': self.knowledge_graph.get_statistics(),
            'agents': {}
        }

        # Get agent-specific metrics
        agents = [
            ('ingestion', self.ingestion_agent),
            ('reader', self.reader_agent),
            ('summarizer', self.summarizer_agent),
            ('entity_extraction', self.entity_extraction_agent),
            ('relationship_extraction', self.relationship_extraction_agent),
            ('schema_alignment', self.schema_alignment_agent),
            ('conflict_resolution', self.conflict_resolution_agent),
            ('evaluator', self.evaluator_agent)
        ]

        for name, agent in agents:
            stats['agents'][name] = agent.get_metrics()

        return stats

    def process_batch(
        self,
        sources: List[Union[str, Path]],
        domain: str = "biomedical",
        relevance_threshold: float = 0.2
    ) -> List[IntermediateOutput]:
        """
        Process multiple documents in batch.

        Args:
            sources: List of document sources
            domain: Domain context
            relevance_threshold: Minimum relevance threshold

        Returns:
            List of processing results
        """
        results = []

        for i, source in enumerate(sources):
            self._log(f"Processing document {i+1}/{len(sources)}: {source}")
            try:
                result = self.process_document(source, domain, relevance_threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {source}: {str(e)}")
                # Create empty result for failed processing
                failed_result = IntermediateOutput()
                failed_result.metrics.error_count = 1
                results.append(failed_result)

        return results