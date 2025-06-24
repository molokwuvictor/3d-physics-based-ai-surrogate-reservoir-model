import os
import glob
import numpy as np
from io import StringIO
from typing import List, Dict, Union

def is_float(s: str) -> bool:
    """
    Return True if the string s can be converted to a float.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_mostly_numbers(line: str, threshold: float = 0.7) -> bool:
    """
    Check if most tokens in the line (using tab-splitting) can be interpreted as floats.
    This preserves empty tokens so that the column structure is not lost.
    """
    # Split by tab and consider only nonempty tokens.
    tokens = [tok.strip() for tok in line.split("\t") if tok.strip() != ""]
    if not tokens:
        return False
    num_numeric = sum(1 for t in tokens if is_float(t))
    return num_numeric / len(tokens) >= threshold

def merge_header_lines(header_lines: List[str]) -> List[str]:
    """
    Merges multiple header lines into fixed columns.
    
    - Uses tab-delimited splitting so that empty columns are preserved.
    - The first header line defines the number of columns.
    - Each subsequent header line is split by tab and, if necessary, padded with empty strings,
      then its tokens are appended to the corresponding column.
    
    Returns:
        A list of strings (one per column) where each string is the merged header text.
    """
    # Use tab-splitting for the first header line.
    first_tokens = [token.strip() for token in header_lines[0].split("\t")]
    ncols = len(first_tokens)
    columns = first_tokens.copy()
    
    for hl in header_lines[1:]:
        tokens = [token.strip() for token in hl.split("\t")]
        # Ensure we have exactly ncols tokens by padding or slicing.
        if len(tokens) < ncols:
            tokens.extend([""] * (ncols - len(tokens)))
        elif len(tokens) > ncols:
            tokens = tokens[:ncols]
        for i in range(ncols):
            if tokens[i]:
                columns[i] += " " + tokens[i]
    # Remove excess whitespace from each merged column.
    return [col.strip() for col in columns]

def parse_tabular_file_from_string(
    data_str: str,
    target_spec: Union[Dict[str, List[str]], List[Union[str, List[str]]]]
) -> Dict[str, np.ndarray]:
    """
    Parses tabular data from a string that may contain multiple segmented tables.
    
    Each table is assumed to be separated by blank lines. For each table, a header block is collected
    (skipping lines starting with "SUMMARY" and lines that are blank). The header block may span multiple lines.
    These header lines are merged column-wise via tab splitting (which preserves empty columns).
    
    The target_spec can be given as:
      - A dictionary mapping keys to a list of required substrings (e.g., { "COPR": ["COPR", "15 15 1"], "WBHP": ["WBHP"] })
      - Or a list where each element is either a string (e.g. "WBHP") or a list of strings (e.g. ["COPR", "15 15 1"]).
        In this case, the first element of a sub-list is used as the key.
    
    For each table, after merging its header, each target key is matched by normalizing the merged header (i.e.
    splitting it and re-joining with a single space) and then checking if every required substring (also normalized)
    appears. Data rows are then read (lines that are mostly numeric using tab-splitting), and the value from the
    matching column is extracted and converted to float.
    
    If the same key occurs in multiple segmented tables, the values are concatenated.
    
    Returns:
        A dictionary mapping each target key to a NumPy array of extracted float values.
        If a key is not found in any table, its value is None.
    """
    # Convert target_spec to dictionary form if it is given as a list.
    if isinstance(target_spec, list):
        target_dict = {}
        for item in target_spec:
            if isinstance(item, list) and len(item) > 0:
                key = item[0]
                target_dict[key] = item
            elif isinstance(item, str):
                target_dict[item] = [item]
        target_spec = target_dict

    # Split data into lines.
    lines = data_str.split("\n")

    # Process each line: remove first tab and trailing whitespace
    lines = [line.lstrip("\t").rstrip() for line in lines]
    
    result = {key: [] for key in target_spec}
    i = 0
    n_lines = len(lines)
    
    while i < n_lines:
        # --- Skip leading blank lines and lines starting with "SUMMARY" ---
        while i < n_lines and (not lines[i].strip() or lines[i].strip().upper().startswith("SUMMARY")):
            i += 1
        if i >= n_lines:
            break
        
        # --- Collect header block for current table ---
        header_block = []
        while i < n_lines and lines[i].strip() and not is_mostly_numbers(lines[i]):
            # Also skip lines beginning with "SUMMARY"
            if not lines[i].strip().upper().startswith("SUMMARY"):
                header_block.append(lines[i].strip())
            i += 1
        
        if not header_block:
            continue
        
        # --- Merge header lines column-wise ---
        merged_headers = merge_header_lines(header_block)
        # Normalize each merged header column by replacing multiple spaces with a single space.
        normalized_headers = [' '.join(col.split()) for col in merged_headers]
        # Debug: print("Normalized Headers:", normalized_headers)
        
        # --- Determine target column indices for this table ---
        key_col_map = {}
        for key, phrases in target_spec.items():
            # Normalize each required phrase.
            norm_phrases = [' '.join(phrase.split()) for phrase in phrases]
            for col_idx, col_text in enumerate(normalized_headers):
                # Check that all normalized phrases appear in the normalized header text.
                if all(phrase in col_text for phrase in norm_phrases):
                    key_col_map[key] = col_idx
                    break
        
        # If no target columns were found in this table, skip the data block.
        if not key_col_map:
            while i < n_lines and lines[i].strip():
                i += 1
            continue
        
        # --- Skip any blank lines between header and data rows ---
        while i < n_lines and not lines[i].strip():
            i += 1
        
        # --- Read data rows for the current table ---
        while i < n_lines and lines[i].strip() and is_mostly_numbers(lines[i]):
            # Use tab-splitting for data rows.
            tokens = [token.strip() for token in lines[i].split("\t")]
            for key, col_idx in key_col_map.items():
                if col_idx < len(tokens) and tokens[col_idx]:
                    try:
                        result[key].append(float(tokens[col_idx]))
                    except ValueError:
                        result[key].append(np.nan)
            i += 1
        
        # --- Skip blank lines between tables ---
        while i < n_lines and not lines[i].strip():
            i += 1

    # --- Convert collected data lists to NumPy arrays (or None if empty) ---
    for key in result:
        result[key] = np.array(result[key]) if result[key] else None

    return result

def parse_files_from_directory(directory: str, extension: str, target_spec: Union[Dict[str, List[str]], List[Union[str, List[str]]]]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Reads and parses all files with the given extension from the specified directory.
    
    Each file is processed using parse_tabular_file_from_string(), and the results are stored in a dictionary
    mapping filenames to their respective parse result (a dictionary mapping target keys to NumPy arrays).
    """
    results = {}
    pattern = os.path.join(directory, "*" + extension)
    files = glob.glob(pattern)
    for i, file_path in enumerate(files):
        with open(file_path, "r") as f:
            content = f.read()
        parsed_result = parse_tabular_file_from_string(content, target_spec)
        # results[os.path.basename(file_path)] = parsed_result
        results[str(i)] = parsed_result
    return results

# --- Example Testing Section ---

if __name__ == "__main__":
    # Example test data with two segmented tables.
    test_data = (
        # --- Table 1: Contains COPR data ---
        " SUMMARY OF RUN 2D_GAS_CONDBI_UNIP_SQUB_3Y_SEN_6\n"
        " TIME        	YEARS       	CGPR        	COPR        	CWPR        	CPR         	CGPRL       	COPRL       	CWPRL       	CPRL        \n"
        " DAYS        	YEARS       	MSCF/DAY    	STB/DAY     	STB/DAY     	PSIA        	MSCF/DAY    	STB/DAY     	STB/DAY     	PSIA       \n"
        "            	            WELL 1      	WELL 1      	WELL 1      	WELL 1      	WELL 1      	WELL 1      	WELL 1      	WELL 1     \n"
        "            	            15 15  1    	15 15  1    	15 15  1    	15 15  1    	15 15  1    	15 15  1    	15 15  1    	15 15  1   \n"
        " 1.000000  	  0.002738  	  35000.00  	  3800.776  	         0  	  1887.859  	  35000.00  	  3800.776  	         0  	  1887.859  	\n"
        " 2.000000  	  0.005476  	  35000.00  	  3800.776  	         0  	  1774.440  	  35000.00  	  3800.776  	         0  	  1774.440  	\n"
        " 3.000000  	  0.008214  	  35000.00  	  3800.776  	         0  	  1718.626  	  35000.00  	  3800.776  	         0  	  1718.626  	\n"
        " 4.000000  	  0.010951  	  35000.00  	  3800.776  	         0  	  1681.117  	  35000.00  	  3800.776  	         0  	  1681.117  	\n"
        " 5.000000  	  0.013689  	  35000.00  	  3800.776  	         0  	  1652.405  	  35000.00  	  3800.776  	         0  	  1652.405  	\n"
        " 6.000000  	  0.016427  	  35000.00  	  3800.776  	         0  	  1628.678  	  35000.00  	  3800.776  	         0  	  1628.678  	\n"
        " 7.000000  	  0.019165  	  35000.00  	  3800.776  	         0  	  1607.980  	  35000.00  	  3800.776  	         0  	  1607.980  	\n"
        " 8.000000  	  0.021903  	  35000.00  	  3800.776  	         0  	  1589.318  	  35000.00  	  3800.776  	         0  	  1589.318  	\n"
        " 9.000000  	  0.024641  	  35000.00  	  3800.776  	         0  	  1571.994  	  35000.00  	  3800.776  	         0  	  1571.994  	\n"
        "\n"  # Blank line separating tables.
        # --- Table 2: Contains WBHP data ---
        " SUMMARY OF RUN 2D_GAS_CONDBI_UNIP_SQUB_3Y_SEN_6\n"
        " TIME        	FPR         	WOGR        	WGOR        	WWPR        	WOPR        	WGPR        	WBHP            \n"
        " DAYS        	PSIA        	STB/MSCF    	MSCF/STB    	STB/DAY     	STB/DAY     	MSCF/DAY    	PSIA            \n"
        "            	            WELL 1      	WELL 1      	WELL 1      	WELL 1      	WELL 1      	WELL 1           \n"
        "            	           	           	           	           	           	           	                \n"
        " 1.000000  	  4984.376  	  0.108594  	  9.208646  	         0  	  3800.776  	  35000.00  	  1887.859      \n"
        " 2.000000  	  4968.824  	  0.108594  	  9.208646  	         0  	  3800.776  	  35000.00  	  1774.440      \n"
        " 3.000000  	  4953.284  	  0.108594  	  9.208646  	         0  	  3800.776  	  35000.00  	  1718.626      \n"
        " 4.000000  	  4937.760  	  0.108594  	  9.208646  	         0  	  3800.776  	  35000.00  	  1681.117      \n"
        " 5.000000  	  4922.262  	  0.108594  	  9.208646  	         0  	  3800.776  	  35000.00  	  1652.405      \n"
        " 6.000000  	  4906.796  	  0.108594  	  9.208646  	         0  	  3800.776  	  35000.00  	  1628.678      \n"
        " 7.000000  	  4891.370  	  0.108594  	  9.208646  	         0  	  3800.776  	  35000.00  	  1607.980      \n"
        " 8.000000  	  4875.987  	  0.108594  	  9.208646  	         0  	  3800.776  	  35000.00  	  1589.318      \n"
        " 9.000000  	  4860.653  	  0.108594  	  9.208646  	         0  	  3800.776  	  35000.00  	  1571.994      \n"
    )
    
    # Target specification defined as a list of keys:
    # First key: ['COPR', '15 15 1'] means we require both substrings in the merged header.
    # Second key: ['WBHP'] means we look for a column containing "WBHP".
    target_spec = [['TIME'],['COPR', '15 15 1'], ['WOPR','WELL 1'], ['WWPR', 'WELL 1']]
    
    parsed_result = parse_tabular_file_from_string(test_data, target_spec)
    
    print("COPR values from test data:")
    print(parsed_result.get("COPR"))
    print("WBHP values from test data:")
    print(parsed_result.get("WBHP"))
    
    # To test reading files from a directory, adjust the path and extension below:
    directory_path = r'C:\Users\User\Documents\PHD_HW_Machine_Learning_old\ML_Cases\Training_Data\Simulation_Data\Gas_Cond_BiComp_UniProp_Square_Boundary_2\Well_Files'
    results_from_dir = parse_files_from_directory(directory_path, ".RSM", target_spec)
    #print("Parsed results from directory:", results_from_dir)

