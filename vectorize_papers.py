import os
from dotenv import load_dotenv
from supabase import create_client, Client
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Load environment variables
load_dotenv(override=True)

# Initialize Supabase clientss
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
supabase: Client = create_client(supabase_url, supabase_key)

print(f"Supabase URL: {supabase_url}")
print(f"Supabase Key (first 10 characters): {supabase_key[:10]}...")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY must be set in environment variables")
pc = Pinecone(api_key=pinecone_api_key)

# Create Pinecone index if it doesn't exist
index_name = "research-articles"
dimension = 384
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-west-2')  # Adjust cloud and region as needed
    )
index = pc.Index(index_name)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_articles_from_supabase():
    try:
        all_articles = []
        page = 0
        page_size = 100  # Adjust this value if needed
        total_count = verify_article_count()  # Get the total count of articles

        while len(all_articles) < total_count:
            start_range = page * page_size
            end_range = (page + 1) * page_size - 1
            print(f"Fetching range: {start_range} to {end_range}")

            response = supabase.table("Article").select(
                "id", "doi", "originalPaperTitle", "authorName", "title", "subtitle",
                "simpleSummary", "defaultSummary", "oneCardSummary"
            ).range(start_range, end_range).execute()
            
            articles = response.data
            all_articles.extend(articles)
            
            print(f"Retrieved {len(articles)} articles from page {page + 1}")
            print(f"Total articles retrieved so far: {len(all_articles)}")
            
            if len(articles) == 0:
                print("No more articles retrieved, breaking the loop")
                break
            
            page += 1

        print(f"Total articles retrieved from Supabase: {len(all_articles)}")
        print(f"Expected total articles: {total_count}")
        
        if len(all_articles) != total_count:
            print(f"Warning: Number of retrieved articles ({len(all_articles)}) does not match the total count ({total_count})")
        
        # Log information about the first and last few articles
        for i, article in enumerate(all_articles[:3] + all_articles[-3:]):
            print(f"\nArticle {i + 1 if i < 3 else len(all_articles) - 2 + i % 3}:")
            for key, value in article.items():
                if key in ['simpleSummary', 'defaultSummary', 'oneCardSummary']:
                    print(f"  {key}: {type(value)}")
                else:
                    print(f"  {key}: {value[:50]}..." if isinstance(value, str) and len(value) > 50 else f"  {key}: {value}")
        
        return all_articles
    except Exception as e:
        print(f"Error extracting articles from Supabase: {e}")
        raise
    
def verify_article_count():
    try:
        response = supabase.table("Article").select("id", count="exact").execute()
        total_count = response.count
        print(f"Total number of articles in Supabase (from count query): {total_count}")
        return total_count
    except Exception as e:
        print(f"Error verifying article count: {e}")
        raise

def generate_embedding(text):
    return model.encode(text)

import json

def process_articles(articles):
    processed_articles = []
    skipped_articles = 0
    for article in articles:
        try:
            # Combine relevant fields for embedding
            content_to_embed = f"{article.get('title', '')} {article.get('subtitle', '')} {article.get('originalPaperTitle', '')} {article.get('authorName', '')}"
            
            # Add summaries if they exist and are not null
            for summary_type in ['simpleSummary', 'defaultSummary', 'oneCardSummary']:
                summary = article.get(summary_type)
                if summary:
                    if isinstance(summary, str):
                        try:
                            summary = json.loads(summary)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse {summary_type} for article {article.get('id', 'unknown')}")
                            continue
                    if isinstance(summary, dict):
                        content_to_embed += f" {summary.get('summary', '')}"
                    elif isinstance(summary, list) and len(summary) > 0:
                        if isinstance(summary[0], dict):
                            content_to_embed += f" {summary[0].get('summary', '')}"
                        elif isinstance(summary[0], str):
                            content_to_embed += f" {summary[0]}"

            if not content_to_embed.strip():
                print(f"Warning: No content to embed for article {article.get('id', 'unknown')}")
                skipped_articles += 1
                continue

            embedding = generate_embedding(content_to_embed)
            processed_articles.append({
                'id': str(article.get('id', '')),
                'values': embedding.tolist(),
                'metadata': {
                    'doi': article.get('doi', ''),
                    'title': article.get('title', '')[:100],  # Truncate title if too long
                    'authorName': article.get('authorName', '')[:100]  # Truncate author name if too long
                }
            })
        except Exception as e:
            print(f"Error processing article {article.get('id', 'unknown')}: {str(e)}")
            skipped_articles += 1

    print(f"Processed {len(processed_articles)} articles")
    print(f"Skipped {skipped_articles} articles")
    return processed_articles

def upload_to_pinecone(processed_articles, batch_size=100):
    total_articles = len(processed_articles)
    for i in range(0, total_articles, batch_size):
        batch = processed_articles[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"Uploaded {min(i+batch_size, total_articles)}/{total_articles} articles to Pinecone")
        except Exception as e:
            print(f"Error uploading batch to Pinecone: {e}")
            print("Attempting to upload articles one by one...")
            for article in batch:
                try:
                    index.upsert(vectors=[article])
                    print(f"Successfully uploaded article {article['id']}")
                except Exception as inner_e:
                    print(f"Failed to upload article {article['id']}: {str(inner_e)}")
                    
def reset_pinecone_index():
    try:
        # Delete all vectors in the index
        index.delete(delete_all=True)
        print("All vectors deleted from Pinecone index.")
        
        # Verify that the index is empty
        stats = index.describe_index_stats()
        if stats['total_vector_count'] == 0:
            print("Pinecone index reset successful. The index is now empty.")
        else:
            print(f"Warning: Index reset may have failed. There are still {stats['total_vector_count']} vectors in the index.")
    except Exception as e:
        print(f"Error resetting Pinecone index: {e}")
        raise

def main():
    try:
        print("Resetting Pinecone index...")
        reset_pinecone_index()

        print("\nVerifying total article count in Supabase...")
        total_count = verify_article_count()

        print("\nExtracting articles from Supabase...")
        articles = extract_articles_from_supabase()
        print(f"Extracted {len(articles)} articles")

        if len(articles) != total_count:
            print(f"Warning: Number of extracted articles ({len(articles)}) does not match the total count ({total_count})")

        print("\nProcessing articles and generating embeddings...")
        processed_articles = process_articles(articles)

        print(f"\nSuccessfully processed {len(processed_articles)} articles")

        print("\nUploading articles to Pinecone...")
        upload_to_pinecone(processed_articles)

        print("\nVectorization process complete!")
        print(f"Total articles in Supabase: {total_count}")
        print(f"Total articles extracted: {len(articles)}")
        print(f"Total articles vectorized and uploaded to Pinecone: {len(processed_articles)}")
        print(f"Articles skipped: {len(articles) - len(processed_articles)}")

        # Verify final count in Pinecone
        final_stats = index.describe_index_stats()
        print(f"\nFinal vector count in Pinecone: {final_stats['total_vector_count']}")
    except Exception as e:
        print(f"An error occurred during the vectorization process: {e}")

if __name__ == "__main__":
    main()
    