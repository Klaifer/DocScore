# Code
Clone this repository and check the src/README.md file


# Metrics
Complete result for Rouge-1/2/L metrics for the evaluated datasets.

## Rouge-1
|             | Multi-News |      WCEP |  XScience |     arXiv |   WikiSum |
|:------------|-----------:|----------:|----------:|----------:|----------:|
| Oracle      |    51.58   |  56.27    |   39.23   |   55.98   |   28.80   |
| Oracle-Lead |      48.19 |     49.05 |     34.08 |     46.03 |     40.82 |
| LeadSum     |      44.00 |     30.55 |     29.22 |     35.39 |     20.73 |
| LeadSum_d   |  **45.84** | **35.13** | **30.43** | **44.12** | **29.87** |
| TextRank    |      42.56 | **32.28** | **31.46** |     39.76 |     28.02 |
| TextRank_d  |  **44.89** | **32.98** | **31.33** | **43.09** | **31.11** |
| PacSum      |      44.29 |     30.54 |     30.33 |     38.36 |     21.72 |
| PacSum_d    |  **45.87** | **32.74** | **30.56** | **44.07** | **28.96** |
| BertSum     |      44.32 |     29.97 |     31.61 |     35.74 |     26.73 |
| BertSum_d   |  **45.60** | **34.83** | **32.04** | **42.93** | **32.96** |


## Rouge-2
|             | Multi-News |      WCEP |  XScience |     arXiv |   WikiSum |
|:--------------|----------:|----------:|---------:|----------:|----------:|
| Oracle        |     24.76 |     29.04 |     9.20 |     23.88 |     13.65 |
| Oracle-Lead   |     18.86 |     23.83 |     7.03 |     17.78 |     21.14 |
| LeadSum       |     14.91 |      9.89 |     4.69 |      9.57 |      4.40 |
| LeadSum_d     | **16.39** | **12.37** | **5.17** | **16.41** | **10.93** |
| TextRank      |     13.72 | **11.07** | **5.98** |     13.02 |     10.69 |
| TextRank_d    | **15.35** | **11.34** | **5.86** | **15.87** | **12.33** |
| PacSum        |     14.90 |      9.78 |     5.08 |     10.94 |      4.84 |
| PacSum_d      | **16.32** | **11.00** | **5.16** | **16.20** | **10.11** |
| BertSum       |     14.89 |      9.40 |     5.64 |      9.50 |      8.56 |
| BertSum_d     | **15.89** | **12.13** | **5.87** | **14.98** | **14.29** |

## Rouge-L
|             | Multi-News |       WCEP |    XScience |     arXiv |   WikiSum |
|:------------|-----------:|-----------:|------------:|----------:|----------:|
| Oracle      |      46.59 |      45.65 |       32.83 |     47.82 |     24.12 |
| Oracle-Lead |      44.04 |      39.52 |       29.77 |     40.78 |     35.07 |
| LeadSum     |      40.02 |      23.78 |       25.66 |     31.25 |     17.23 |
| LeadSum_d   |  **41.81** |  **27.14** |   **26.73** | **39.14** | **25.21** |
| TextRank    |      38.23 |  **24.76** |   **27.36** |     34.06 |     23.70 |
| TextRank_d  |  **40.66** |  **25.13** |   **27.27** | **37.49** | **26.15** |
| PacSum      |      40.29 |      23.72 |       26.24 |     33.45 |     17.90 |
| PacSum_d    |  **41.83** |  **25.20** |   **26.48** | **38.95** | **24.22** |
| BertSum     |      40.13 |      23.24 |       27.53 |     31.44 |     22.61 |
| BertSum_d   |  **41.40** |  **26.80** |   **27.92** | **37.83** | **28.37** |


# Citation

If you find this content useful, please cite:
```
@INPROCEEDINGS{garcia2024siamese,
    AUTHOR="Klaifer Garcia and Lilian Berton",
    TITLE="Siamese Network-Based Prioritization for Enhanced Multi-Document Summarization",
    BOOKTITLE="BRACIS 2024 () ",
    ADDRESS="",
    DAYS="23-21",
    MONTH="may",
    YEAR="2024",
    KEYWORDS="- Natural Language Processing; - Neural Networks"
}
```
