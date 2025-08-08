#!/usr/bin/env python3
"""
Pinecone Setup Script for Document Query System
Usage: python pinecone_setup.py [create|delete|info]
"""

import os
import sys
import pinecone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_pinecone():
    """Initialize Pinecone"""
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    pinecone.init(api_key=api_key, environment=environment)
    return True

def create_index(index_name="document-query-system"):
    """Create Pinecone index"""
    try:
        init_pinecone()
        
        # Check if index already exists
        if index_name in pinecone.list_indexes():
            logger.info(f"Index '{index_name}' already exists")
            return
        
        # Create index
        pinecone.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2 dimension
            metric='cosine'
        )
        
        logger.info(f"Successfully created index: {index_name}")
        
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        sys.exit(1)

def delete_index(index_name="document-query-system"):
    """Delete Pinecone index"""
    try:
        init_pinecone()
        
        if index_name not in pinecone.list_indexes():
            logger.info(f"Index '{index_name}' does not exist")
            return
        
        pinecone.delete_index(index_name)
        logger.info(f"Successfully deleted index: {index_name}")
        
    except Exception as e:
        logger.error(f"Error deleting index: {e}")
        sys.exit(1)

def get_index_info(index_name="document-query-system"):
    """Get index information"""
    try:
        init_pinecone()
        
        if index_name not in pinecone.list_indexes():
            logger.info(f"Index '{index_name}' does not exist")
            return
        
        index = pinecone.Index(index_name)
        stats = index.describe_index_stats()
        
        logger.info(f"Index: {index_name}")
        logger.info(f"Total vectors: {stats['total_vector_count']}")
        logger.info(f"Index fullness: {stats['index_fullness']}")
        logger.info(f"Dimension: {stats['dimension']}")
        
        if 'namespaces' in stats:
            logger.info(f"Namespaces: {list(stats['namespaces'].keys())}")
        
    except Exception as e:
        logger.error(f"Error getting index info: {e}")
        sys.exit(1)

def list_indexes():
    """List all indexes"""
    try:
        init_pinecone()
        indexes = pinecone.list_indexes()
        
        logger.info("Available indexes:")
        for idx in indexes:
            logger.info(f"  - {idx}")
        
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python pinecone_setup.py [create|delete|info|list]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    index_name = sys.argv[2] if len(sys.argv) > 2 else "document-query-system"
    
    if command == "create":
        create_index(index_name)
    elif command == "delete":
        delete_index(index_name)
    elif command == "info":
        get_index_info(index_name)
    elif command == "list":
        list_indexes()
    else:
        print("Invalid command. Use: create, delete, info, or list")
        sys.exit(1)

if __name__ == "__main__":
    main()
