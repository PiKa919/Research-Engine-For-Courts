
import os
import subprocess
import shutil

def is_aws_cli_installed():
    """Check if the AWS CLI is installed."""
    return shutil.which("aws") is not None

def download_from_aws():
    """
    Lists the files available in the Indian Supreme Court Judgments dataset on AWS Open Data.
    
    To download the files, you can use the AWS CLI with the following command:
    aws s3 cp --no-sign-request s3://indian-supreme-court-judgments/<file_name> .
    """
    print("Listing files from AWS Open Data...")
    if not is_aws_cli_installed():
        print("AWS CLI not found.")
        print("Please install it by following the instructions at: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")
        return

    command = "aws s3 ls --no-sign-request s3://indian-supreme-court-judgments/"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("Files available for download:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while listing files from AWS: {e}")
        print(f"Stderr: {e.stderr}")

def download_from_kaggle():
    """
    Provides instructions for downloading the Indian Supreme Court Judgments dataset from Kaggle.
    """
    print("\nTo download the dataset from Kaggle, please follow these steps:")
    print("1. Go to: https://www.kaggle.com/datasets/vangap/indian-supreme-court-judgments")
    print("2. Log in to your Kaggle account.")
    print("3. Click on the 'Download' button to download the dataset.")

if __name__ == "__main__":
    print("Starting data collection...")
    download_from_aws()
    download_from_kaggle()
