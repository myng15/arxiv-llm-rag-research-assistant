import sys
import os.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import os
from config import data_dir
from tqdm import tqdm
import pandas as pd

def process_data(processed_data_path=os.path.join(data_dir, "arxiv_processed.csv")):

    data_path = os.path.join(data_dir, "arxiv-metadata-oai-snapshot.json")
    n_lines = sum(1 for line in open(data_path))
    print("n_lines: ", n_lines)
    batch_size = 10000

    if os.path.exists(processed_data_path):
        print("Data has already been processed.")
        return

    for df in tqdm(pd.read_json(data_path, lines=True, chunksize=batch_size), total=n_lines // batch_size, desc="Processing data", unit="batch"):
        # Set datatype as string
        df = df.astype(str)
        # replace nan with empty string
        df = df.replace("nan", "")

        # All the features:
        # ['id', 'submitter', 'authors', 'title', 'comments', 'journal-ref', 'doi',
        #'report-no', 'categories', 'license', 'abstract', 'versions',
        # 'update_date', 'authors_parsed']
        #
        # Filter necessary features
        df = df[["id", "abstract", "title", "doi", "categories", "update_date", "authors_parsed"]]

        # Remove rows with empty abstracts, titles, or ids
        df = df[(df["abstract"] != "") & (df["title"] != "") & (df["id"] != "")]
        
        # Filter to include only AI-related papers
        topics = ['cs.AI', 'cs.CV', 'cs.IR', 'cs.LG', 'cs.CL']

        # Create a regular expression pattern that matches any of the topics
        # The pattern will look like 'cs.AI|cs.CV|cs.IR|cs.LG|cs.CL'
        pattern = '|'.join(topics)

        # Filter the DataFrame to include rows where the 'categories' column contains any of the topics
        # na=False to make sure that NaN values are treated as False
        df_filtered = df[df['categories'].str.contains(pattern, na=False)]

        # To csv
        df_filtered.to_csv(processed_data_path, mode="a", header=True if not os.path.exists(processed_data_path) else False, index=False)


def analyze_processed_data(processed_data_path=os.path.join(data_dir, "arxiv_processed.csv")):
    if not os.path.exists(processed_data_path):
        print("Processed data file not found!")
        return
    
    print("Loading processed data...")
    df = pd.read_csv(processed_data_path, usecols=["categories", "update_date"])  # Load only needed columns
    
    print(f"Total records: {df.shape[0]}")  # Print dataset size

    # Get unique categories
    unique_categories = set()
    df["categories"].dropna().str.split().apply(unique_categories.update)  

    print(f"Unique categories found ({len(unique_categories)}):")
    print(sorted(unique_categories))  

    # Define required topics
    required_topics = {'cs.AI', 'cs.CV', 'cs.IR', 'cs.LG', 'cs.CL'}

    # Check if every row has at least one required topic
    all_valid = df["categories"].dropna().apply(lambda x: any(topic in x.split() for topic in required_topics)).all()

    if all_valid:
        print("All rows contain at least one required topic.")
    else:
        print("Some rows do not contain the required topics!")


    # Convert update_date to datetime for min/max operations
    df["update_date"] = pd.to_datetime(df["update_date"], errors="coerce")  # Handle errors gracefully
    oldest_date = df["update_date"].min()
    latest_date = df["update_date"].max()

    print(f"Oldest update date: {oldest_date}")
    print(f"Latest update date: {latest_date}")


if __name__ == "__main__":
    print("Processing data..")
    process_data()
    print("Processing: Done!")

    #analyze_processed_data()








