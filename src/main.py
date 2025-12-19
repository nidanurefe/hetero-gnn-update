from logging_config import setup_logging, get_logger

# DATA
from preprocess import main as preprocess
from build_graph import main as build_graph
from build_graph_target_only import main as build_graph_target_only

# TRAINING
from train_gat_neighbor import main as train_gat
from train_sage_neighbor import main as train_sage

# EVALUATION
from eval_rank import main as eval_rank


if __name__ == "__main__":
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting pipeline...")

    logger.info("Preprocessing raw datasets...")
    preprocess()

    logger.info("Building hetero graph...")
    build_graph()

    logger.info("Training GAT model...")
    train_gat()

    # logger.info("Training GraphSAGE model...")
    # train_sage()

    logger.info("Evaluating GAT model...")
    eval_rank()
    
    logger.info("Pipeline completed successfully!")