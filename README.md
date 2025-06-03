# T-Test Analysis Application

A Streamlit web application for performing statistical t-tests with comprehensive analysis and visualization.

## Created By
Galuh Adi Insani  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/)

## Features

- Support for both One Sample T-Test and Independent T-Test
- Comprehensive statistical analysis including:
  - T-test statistics and p-values
  - Effect size calculation (Cohen's d)
  - Confidence intervals
  - Descriptive statistics
  - ANOVA analysis (for Independent T-Test)
- Enhanced data visualizations:
  - Distribution plots
  - Box plots
  - Q-Q plots
  - Violin plots (for Independent T-Test)
- Detailed interpretation of results
- Support for CSV and Excel file inputs

## Installation

1. Clone this repository or download the files
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run ttest.py
```

2. Access the application in your web browser (typically at http://localhost:8501)

3. Select the type of t-test you want to perform:
   - One Sample T-Test
   - Independent T-Test

4. Upload your data file (CSV or Excel format)
   - For One Sample T-Test: Upload a file with at least one column of numerical data
   - For Independent T-Test: Upload a file with at least two columns of numerical data

5. Configure your analysis:
   - Select the significance level (α)
   - Choose the data columns for analysis
   - For One Sample T-Test, input the hypothesized mean

6. Click "Calculate" to perform the analysis

## Sample Data Files

The repository includes two sample data files:
- `one_sample_test.xlsx`: Example data for One Sample T-Test
- `independent_test.xlsx`: Example data for Independent T-Test

## Output

The application provides:
- Descriptive statistics
- Test results with p-values
- Effect size calculations
- Confidence intervals
- Visual representations of the data
- Detailed interpretations of all statistics
- ANOVA results (for Independent T-Test)

## Requirements

- Python 3.7+
- Streamlit
- NumPy
- SciPy
- Pandas
- Seaborn
- Matplotlib

## License

This project is open source and available under the MIT License.
