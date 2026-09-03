"""
Data loader module for loading datasets in .md format
"""

import re
import pandas as pd
from pathlib import Path

def find_sections(content: str, header_start='#'):
    structure = []
    sections = re.split(rf'^{header_start}\s', content, flags=re.MULTILINE)

    for section in sections[1:]:
        splits = section.split('\n', 1)

        # Define components of the section
        components = dict()
        components['section'] = splits[0].strip()
        components['subsections'] = find_sections(splits[1], header_start='#' + header_start)

        # If no subsections, extract the content as a dict using its key-value structure
        if components['subsections'] == []:
            matches = re.findall(r'\*\*(.*?)\*\*\s*\n\s*(.*?)\n', splits[1])
            components['content'] = {key.strip(): value.strip() for key, value in matches}

        structure.append(components)
    
    return structure

# Convert a QA file in .md format into csv format and save
def read_markdown(file_path, output_path, num_samples=None, ignore_cache=False, encoding='utf-8'):

    qa_output_path = Path(output_path)

    if not ignore_cache and qa_output_path.exists():
        return pd.read_csv(qa_output_path)
    
    qa_input_path = Path(file_path)

    if not qa_input_path.exists():
        raise FileNotFoundError(f"File not found: {qa_input_path}")

    with open(qa_input_path, 'r', encoding=encoding) as file:
        content = file.read()
    
    json = find_sections(content)

    data_samples = []
    for row in json[0]['subsections']:
        # Use map function here or lambda function
        old_keys = list(row['content'].keys())
        for key in old_keys:
            new_key = key.lower().replace(' ', '_').replace(':', '')
            row['content'][new_key] = row['content'].pop(key)
        data_samples.append(row['content'])
    
    df = pd.DataFrame(data_samples)

    if num_samples is not None:
        df = df.head(num_samples)

    df.to_csv(output_path, index=False)

    return df
