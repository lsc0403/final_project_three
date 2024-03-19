import json

import pandas as pd


def excel_to_json_by_movie(file_path, json_file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Initialize an empty dictionary to store the JSON data
    json_data = {}

    # Iterate through each group and convert each group into a list of dictionaries
    for movie_name, group in df.groupby('Movie Name'):
        json_data[movie_name] = group.drop('Movie Name', axis=1).to_dict('records')

    # Convert the dictionary to JSON format
    json_output = json.dumps(json_data, indent=4)

    # Save the JSON data to a file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_output)

# Specify the file paths
# excel_file_path = 'BThree-Modified_IMDB_Reviews.xlsx'  # Update this to your Excel file path
# output_json_file_path = 'BThree-JSONData.json'  # Update this to your desired JSON file path
excel_file_path = 'Excel_File/BThree-Modified_IMDB_Reviews.xlsx'  # Update this to your Excel file path
output_json_file_path = 'Page_and_JSON/BThree-JSONData.json'  # Update this to your desired JSON file path
# Call the function
excel_to_json_by_movie(excel_file_path, output_json_file_path)

print(f'JSON data has been saved to {output_json_file_path}')


