# Fine-Tuning LLM for Test Case Step Generation

## 🧠 Project Goal

This project aims to fine-tune a large language model (LLM) to automatically generate **test case steps** from structured test case metadata.

---

## 🧰 Project Structure

```
LLM-FineTuning-Project/
├── venv/                    # Python virtual environment
├── data/
│   ├── raw_dataset/         # Original JSON files
│   ├── flattened_testcases.csv  # CSV with all test cases flattened
│   └── finetune_dataset.jsonl   # Final training-ready file
├── src/
│   ├── json_to_df.py        # Script to flatten JSONs into a DataFrame
│   ├── build_dataset.py     # Script to convert cleaned data to jsonl
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

---

## ✅ Current Progress

1. **Set up virtual environment** ✅
2. **Flattened JSON files into a single CSV** ✅
3. **Filtered and cleaned columns** ✅
4. **Selected target: `test case steps`** ✅
5. **Identified usable features as input** ✅

---

## 🎯 Fine-Tuning Dataset Format

We use the `.jsonl` format for training:

```json
{
  "input": "Description: CheckNoInhibitions\nParameters: mode, speed\n...",
  "output": "1. Set mode to AUTO\n2. Verify flags..."
}
```

Generated using `build_dataset.py` script from cleaned CSV.

---

## 🔁 Workflow Modularity

* Each transformation step is in its own script (`src/` folder)
* Reusable cached DataFrames are used to avoid reloading files
* Designed to scale with more data / columns easily

---


### 🗓️ Day-wise Updates

**📅 Day 1:**

* Setup environment and received data in yaml files.
* Converted yaml files to json for better readability.
* Flattened all test cases into a single CSV [one level flattenings]

**📅 Day 2:**

* Identified folder-type entries via `SummaryCategories`=='Folder'
* Understood data on colum level
* Removed irrlavent columns from findings.

**📅 Day 3:**

* Trying to flatten remaining columns fro better understanding.
* Flattened ParameterList column
* Added file_name and ParameterList to a seperate data frame where ParameterList != NAN
* ParameterList(string)-->ParameterList(python object) and then normalised

**📅 Day 4:**

* Loaded required in data frame
* Found NAN even after filtering NAN values.
* Error Analysis of the same found the cause for this (Bracket/Parenthesis mismatch)
* Appended Brackets for required files.

**📅 Day 5:**

* Identified columns in ParametersList to remove (commonname, longshortname, labeltype) and
optional removes (referencenumber, reloadflag).
* Dropped the identified columns from the flattened DataFrame.
* Grouped cleaned parameter rows by file_name into list-of-dicts format to restore structure.
* Merged the cleaned parameter list back into the original DataFrame.
* Saved the updated DataFrame to CSV for verification.
* Refactored the code into a modular structure by creating a separate utility 
  file containing reusable components for:
	Parsing and normalising nested columns
	Cleaning unwanted or empty columns
	Merging processed data back to the original DataFrame
	Exporting results for review

**📅 Day 6:**

* Cleaned Parameters, ParameterValues, and Relationships columns.
* Applied same parsing, normalization, and merging techniques as used for ParametersList.
* Ensured final DataFrame structure is consistent for downstream processing.
* Found html tags in pre,post conditions and expected results.

**📅 Day 7:**

* Explored new BMW dataset structure and contents.
* Inspected PreCondition and PostCondition columns for embedded HTML.
* Used regex to identify unique HTML tags present in these fields.
* Cleaned HTML tags from the dataset using BeautifulSoup (bs4).
* Ensured cleaned text retains original meaning and formatting where necessary.

**📅 Day 8:**

* Identified numeric-like HTML tags in TestCaseFormID (regex flagged them as tags).
* Used BeautifulSoup to clean HTML tags in TestCaseFormID, which contains PreConditions, PostConditions, and actual test steps.
* Planned to split these components and re-clean unnecessary fields.

**📅 Day 9:**

commit: refactor(utils): reorganize utility modules and add preprocessing steps
------------------------------------------------------------------------------
* Consolidated parse_column, normalize_column, clean_column, merge_df into columns.py
* Renamed extract_html.py → html.py for clarity
* Updated utils/__init__.py exports accordingly
* Reduced fragmentation by avoiding too many single-function utils files
* ded preprocessing in clean_column to drop:
  • type == "Pre Conditions"
  • type == "Post Conditions"
  • action == "Test Sequence"


**📅 Day 10:**

* Worked on BMW Mite dataset (test cases).
* Identified irrelevant fields to drop (section, reference mode, Document ID, External Script Name, metadata fields).
* Found key useful fields:
    -text (test description),
    -validates (links to requirements, m:m relationship),
    -priority,
    -pre/post conditions (with referenced IDs).
* Confirmed traceability exists: every test case validates at least one requirement.
* Discussed strategy for linking requirements & test cases in fine-tuning dataset:
    -Option A: keep requirement IDs as list inside test case.
    -Decided to go with this option to maintain balance and avoid duplication.
* Considered how to handle pre/post conditions: decided to append referenced ID into description when explicitly mentioned (e.g., “run test case 1059666”).
* For training data, plan to set null for unused fields (to avoid token bloat).


**📅 Day 10:**

* Analyzed BMW Mite requirements dataset.
* Droppable fields: section, Document ID, Legacy Branches, Non-Functional Requirement, Document.
* Useful fields:
    -text (requirement description),
    -Category (Functional/Heading/Interface/Technical/Business).
* Plan: append Heading rows to enrich requirement descriptions for context.
* Confirmed traceability link: validates in test cases ↔ ID in requirements.
* Final mapping ensures every requirement is covered by at least one test case.
* Pipeline refactoring: discussed making pipeline column-agnostic by looping over columns_to_clean list instead of hardcoding.
* Discussed fine-tuning dataset structure: two main files (requirements + test cases) vs merged dataset, leaned toward separate but linked (via validates).







---

## 🛠️ Next Steps

* [ ] Finalize model type (e.g., GPT-3.5, LLaMA, Mistral)
* [ ] Decide format: chat-style or plain prompt ➜ response
* [ ] Perform token length analysis
* [ ] Start fine-tuning using selected framework

---

## 📌 Notes

* DataFrame exploration and cleaning is done using `pandas`
* Prompt format is human-readable and LLM-friendly
* Target output is expected to be multiline, clear instructions

---

Feel free to edit this README as the project evolves!
