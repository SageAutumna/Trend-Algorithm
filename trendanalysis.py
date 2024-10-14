import os
import boto3
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from supabase import create_client, Client
import json

# Load environment variables first
load_dotenv(override=True)

# Set AWS credentials and region for Bedrock
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

# Create AWS Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
index_name = "research-articles"  # Make sure this matches your actual index name

if not pinecone_api_key or not pinecone_environment:
    raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set in environment variables")

pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)

# Check if the index exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Make sure this matches the dimension of your vectors
        metric='cosine'
    )

index = pc.Index(index_name)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Set the number of clusters
n_clusters = 10

# Initialize K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

def fetch_all_vectors():
    try:
        stats = index.describe_index_stats()
        vector_count = stats['total_vector_count']
        print(f"Total vectors in index according to stats: {vector_count}")

        if vector_count == 0:
            print("Index is empty according to stats.")
            return np.array([]), []

        batch_size = 1000
        all_vectors = []
        all_ids = []

        for i in range(0, vector_count, batch_size):
            try:
                results = index.query(
                    vector=[0.1] * 384,  # Dummy vector
                    top_k=min(batch_size, vector_count - i),
                    include_values=True
                )
                print(f"Queried batch {i//batch_size + 1}: {len(results['matches'])} vectors")
                vectors = [match['values'] for match in results['matches']]
                ids = [match['id'] for match in results['matches']]
                all_vectors.extend(vectors)
                all_ids.extend(ids)
            except Exception as e:
                print(f"Error querying batch {i//batch_size + 1}: {e}")

        print(f"Total vectors fetched: {len(all_vectors)}")
        return np.array(all_vectors), all_ids
    except Exception as e:
        print(f"Error in fetch_all_vectors: {e}")
        return np.array([]), []

def perform_clustering(vectors):
    normalized_vectors = normalize(vectors)
    kmeans.fit(normalized_vectors)
    return kmeans.labels_

def update_pinecone_with_clusters(vector_ids, cluster_labels):
    batch_size = 100
    for i in range(0, len(vector_ids), batch_size):
        batch_ids = vector_ids[i:i+batch_size]
        batch_clusters = cluster_labels[i:i+batch_size]

        for id, cluster in zip(batch_ids, batch_clusters):
            update_metadata = {
                'cluster': int(cluster)
            }
            try:
                index.update(id=id, set_metadata=update_metadata)
                print(f"Updated vector {id} with cluster information.")
            except Exception as e:
                print(f"Error updating vector {id}: {e}")

# Update the cluster_vectors function to handle potential empty results
def cluster_vectors():
    print("Starting clustering process...")
    vectors, vector_ids = fetch_all_vectors()
    print(f"Fetched {len(vectors)} vectors from Pinecone.")

    if len(vectors) == 0:
        print("No vectors found in the index. Skipping clustering.")
        return

    # Remove duplicates
    unique_vectors, unique_indices = np.unique(vectors, axis=0, return_index=True)
    unique_ids = [vector_ids[i] for i in unique_indices]

    print(f"Performing K-means clustering on {len(unique_vectors)} unique vectors...")
    cluster_labels = perform_clustering(unique_vectors)
    print(f"Clustering complete. {n_clusters} clusters created.")

    print("Updating Pinecone with cluster information...")
    update_pinecone_with_clusters(unique_ids, cluster_labels)
    print("Pinecone updated with cluster information.")

    # Print cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))
    print("\nCluster Distribution:")
    for cluster, count in cluster_distribution.items():
        print(f"Cluster {cluster}: {count} vectors")

def query_processing(query, top_k=5):
    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results['matches']

def fetch_full_article_data(article_ids):
    print(f"Fetching full article data for {len(article_ids)} articles...")
    try:
        response = supabase.table("Article").select("*").in_("id", article_ids).execute()
        fetched_articles = {article['id']: article for article in response.data}
        print(f"Successfully fetched {len(fetched_articles)} articles from Supabase.")
        
        # Debug: Print a sample of the fetched data
        if fetched_articles:
            sample_id = next(iter(fetched_articles))
            print(f"Sample article data for ID {sample_id}:")
            for key, value in fetched_articles[sample_id].items():
                print(f"  {key}: {str(value)[:50]}..." if isinstance(value, str) else f"  {key}: {value}")
        
        return fetched_articles
    except Exception as e:
        print(f"Error fetching full article data from Supabase: {e}")
        return {}

def trend_analysis(query, top_k=5):
    matches = query_processing(query, top_k)
    print(f"Query processing returned {len(matches)} matches.")

    article_ids = [match['id'] for match in matches]
    full_articles = fetch_full_article_data(article_ids)

    cluster_counts = {}
    content_to_summarize = f"Query: {query}\n\n"
    relevant_papers = []

    for match in matches:
        article_id = match['id']
        article = full_articles.get(article_id)
        if not article:
            print(f"Article with ID {article_id} not found in Supabase.")
            continue

        cluster = match['metadata'].get('cluster', 'Unknown')
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        content_to_summarize += f"Title: {article.get('title', 'No title')}\n"
        content_to_summarize += f"Author: {article.get('authorName', 'No author')}\n"
        content_to_summarize += f"Relevance: {match['score']}\n"
        content_to_summarize += f"Cluster: {cluster}\n"

        summary = article.get('defaultSummary') or article.get('simpleSummary') or article.get('oneCardSummary') or "No summary available"
        
        if isinstance(summary, list):
            if all(isinstance(item, dict) for item in summary):
                summary = " ".join(item.get('summary', 'No summary available') for item in summary)
            else:
                summary = " ".join(summary)

        content_to_summarize += f"Summary: {summary}\n\n"

        relevant_papers.append({
            "title": article.get('title', 'No title'),
            "author": article.get('authorName', 'No author'),
            "relevance_score": match['score'],
            "cluster": cluster,
            "summary": summary[:200] + "..."  # Truncate summary for brevity
        })

    print(f"Processed {len(relevant_papers)} relevant papers.")
    print(f"Content to summarize (first 500 characters):\n{content_to_summarize[:500]}...")

    # Generate summary using Claude through AWS Bedrock
    prompt = f"""Human: Analyze the following research papers in relation to the given query: "{query}". 
    Provide a comprehensive summary that directly answers the query, highlighting key findings 
    from the most relevant papers. Identify trends or patterns across the papers and clusters.

    {content_to_summarize}

    Cluster distribution:
    {', '.join([f"Cluster {k}: {v} papers" for k, v in cluster_counts.items()])}

    Please structure your response as follows:
    1. Direct answer to the query
    2. Key findings from relevant papers
    3. Identified trends or patterns
    4. Conclusion

    Assistant: Based on the provided research papers and the query "{query}", I will provide a comprehensive summary addressing the key points you requested.

    Human: Great, please proceed with the analysis and summary.

    Assistant:"""

    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-v2",
            contentType="application/json",
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 1000,
                "temperature": 0.5,
            })
        )

        result_body = json.loads(response['body'].read())
        summary = result_body['completion']

        return {
            "summary": summary,
            "cluster_distribution": cluster_counts,
            "relevant_papers": relevant_papers
        }

    except Exception as e:
        print(f"Error with AWS Bedrock API: {e}")
        return None

def user_query(query):
    result = trend_analysis(query)
    if result is None:
        print("No results found. There might have been an error with the API or query processing.")
        return

    print("\nQuery:", query)
    print("\nSummarized Answer:")
    print(result.get('summary', 'No summary available'))

    print("\nRelevant Papers:")
    if 'relevant_papers' in result and result['relevant_papers']:
        for paper in result['relevant_papers']:
            print(f"- Title: {paper['title']}")
            print(f"  Author: {paper['author']}")
            print(f"  Relevance Score: {paper['relevance_score']:.2f}")
            print(f"  Cluster: {paper['cluster']}")
            print(f"  Summary: {paper['summary']}")
            print()
    else:
        print("No relevant papers found in the result.")

    print("\nCluster Distribution:")
    if 'cluster_distribution' in result and result['cluster_distribution']:
        for cluster, count in result['cluster_distribution'].items():
            print(f"Cluster {cluster}: {count} papers")
    else:
        print("No cluster distribution available in the result.")

    print("\nDebug Information:")
    print(f"Result keys: {result.keys()}")
    print(f"Number of relevant papers: {len(result.get('relevant_papers', []))}")
    print(f"Length of summary: {len(result.get('summary', ''))}")
    
def main():
    while True:
        print("\n1. Check index status")
        print("2. List sample vector IDs")
        print("3. Perform clustering")
        print("4. Enter a query")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            check_index_status()
        elif choice == '2':
            list_vector_ids()
        elif choice == '3':
            cluster_vectors()
        elif choice == '4':
            if index.describe_index_stats()['total_vector_count'] == 0:
                print("The index is empty. Please add vectors before querying.")
            else:
                query = input("Enter your research query: ")
                user_query(query)
        elif choice == '5':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def check_index_status():
    try:
        stats = index.describe_index_stats()
        print(f"Index statistics:")
        print(f"Total vectors: {stats['total_vector_count']}")
        print(f"Dimension: {stats['dimension']}")
        print(f"Indexed vectors: {stats.get('indexed_vector_count', 'N/A')}")
        print(f"Namespaces: {', '.join(stats.get('namespaces', {}).keys()) or 'None'}")
    except Exception as e:
        print(f"Error checking index status: {e}")
        print("Please ensure your Pinecone API key and environment variables are correctly set.")

def list_vector_ids(limit=10):
    try:
        stats = index.describe_index_stats()
        vector_count = stats['total_vector_count']
        print(f"Total vectors in index: {vector_count}")

        if vector_count == 0:
            print("Index is empty.")
            return

        # Query the index to get some vector IDs
        results = index.query(
            vector=[0.1] * 384,  # Dummy vector
            top_k=min(limit, vector_count),
            include_values=False
        )
        print(f"Sample vector IDs:")
        for match in results['matches']:
            print(f"  - {match['id']}")
    except Exception as e:
        print(f"Error listing vector IDs: {e}")

if __name__ == "__main__":
    main()