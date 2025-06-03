import streamlit as st
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import base64
from io import BytesIO
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="T-Test Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve design
st.markdown("""
    <style>
        .main > div {
            padding: 2rem;
            border-radius: 0.5rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 0.5rem;
            border-radius: 0.3rem;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .pdf-download-link {
            display: inline-block;
            padding: 0.8rem 1.5rem;
            background-color: #1E88E5;
            color: white !important;
            text-decoration: none;
            border-radius: 0.5rem;
            margin: 1.5rem 0;
            text-align: center;
            transition: all 0.3s ease;
            font-size: 1.1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            width: fit-content;
        }
        .pdf-download-link:hover {
            background-color: #1565C0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transform: translateY(-1px);
        }
        h1 {
            color: #1E88E5;
            text-align: center;
            padding: 1rem;
            margin-bottom: 2rem;
        }
        h3 {
            color: #333;
            padding: 0.5rem 0;
            border-bottom: 2px solid #1E88E5;
            margin-top: 1.5rem;
        }
        .stAlert {
            border-radius: 0.3rem;
            margin: 1rem 0;
        }
        div[data-testid="stFileUploader"] {
            border: 2px dashed #ccc;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: #1E88E5;
        }
        .test-type-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        .test-type-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        .test-type-card.one-sample {
            border-left: 5px solid #1E88E5;
        }
        .test-type-card.independent {
            border-left: 5px solid #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Application Header
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title('📊 T-Test Analysis')
    st.markdown("""
        <p style='text-align: center; color: #666; font-size: 1.2em; margin-bottom: 2rem;'>
            A comprehensive statistical analysis tool for comparing means
        </p>
    """, unsafe_allow_html=True)

# Information Cards about test types
st.markdown("""
<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 2rem 0;'>
    <div class='test-type-card one-sample'>
        <h4 style='color: #1E88E5; margin-bottom: 1rem;'>One Sample T-Test</h4>
        <p><strong>When to use:</strong> Compare a sample mean to a known value</p>
        <ul style='margin: 0.8rem 0 1.2rem 1.2rem;'>
            <li>Testing against a standard or target</li>
            <li>Quality control measurements</li>
            <li>Comparing to historical data</li>
        </ul>
        <p style='margin-top: 1rem;'><strong>Requirements:</strong></p>
        <ul style='margin: 0.8rem 0 0 1.2rem;'>
            <li>Normally distributed data</li>
            <li>Random sampling</li>
            <li>Continuous measurements</li>
        </ul>
    </div>
    <div class='test-type-card independent'>
        <h4 style='color: #4CAF50; margin-bottom: 1rem;'>Independent T-Test</h4>
        <p><strong>When to use:</strong> Compare means between two independent groups</p>
        <ul style='margin: 0.8rem 0 1.2rem 1.2rem;'>
            <li>Comparing two different methods</li>
            <li>Treatment vs Control studies</li>
            <li>Before vs After analysis</li>
        </ul>
        <p style='margin-top: 1rem;'><strong>Requirements:</strong></p>
        <ul style='margin: 0.8rem 0 0 1.2rem;'>
            <li>Independent samples</li>
            <li>Normal distribution</li>
            <li>Similar variances</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Helper functions
def calculate_effect_size(data1, data2=None, test_type='one_sample'):
    """Calculate Cohen's d effect size."""
    if test_type == 'one_sample':
        return abs((np.mean(data1) - 0) / np.std(data1, ddof=1))
    
    pooled_std = np.sqrt(((len(data1) - 1) * np.std(data1, ddof=1)**2 + 
                         (len(data2) - 1) * np.std(data2, ddof=1)**2) / 
                        (len(data1) + len(data2) - 2))
    return abs(np.mean(data1) - np.mean(data2)) / pooled_std

def plot_data(data1, data2=None, test_type='one_sample'):
    """Create visualization for the data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if test_type == 'one_sample':
        sns.histplot(data1, kde=True, ax=ax)
        ax.axvline(np.mean(data1), color='red', linestyle='--', label='Sample Mean')
        ax.axvline(0, color='green', linestyle='--', label='Hypothesized Mean')
        ax.legend()
    else:
        data = pd.DataFrame({
            'Group 1': data1,
            'Group 2': data2
        }).melt()
        sns.boxplot(x='variable', y='value', data=data, ax=ax)
        sns.stripplot(x='variable', y='value', data=data, color='red', alpha=0.3, ax=ax)
    
    ax.set_title('Data Distribution')
    plt.tight_layout()
    return fig

def read_file_data(uploaded_file):
    """Read and validate input data file."""
    if uploaded_file is None:
        return None
        
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_type in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
        
        if df.empty:
            raise ValueError("The uploaded file is empty.")
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def interpret_results(results, alpha, test_type):
    """Generate detailed interpretation of test results."""
    interpretation = []
    
    # Significance interpretation
    p_value = results.get('P-value')
    if p_value < alpha:
        interpretation.append(f"✓ Statistical Significance: The results are statistically significant (p = {p_value:.4f} < α = {alpha}).")
        if test_type == 'one_sample':
            interpretation.append("   This suggests strong evidence to reject the null hypothesis that the sample mean equals the hypothesized mean.")
        else:
            interpretation.append("   This suggests strong evidence to reject the null hypothesis that the two group means are equal.")
    else:
        interpretation.append(f"× Statistical Significance: The results are not statistically significant (p = {p_value:.4f} ≥ α = {alpha}).")
        if test_type == 'one_sample':
            interpretation.append("   There is insufficient evidence to reject the null hypothesis that the sample mean equals the hypothesized mean.")
        else:
            interpretation.append("   There is insufficient evidence to reject the null hypothesis that the two group means are equal.")
    
    # Effect size interpretation
    effect_size = results.get('Effect Size (Cohen\'s d)')
    if effect_size < 0.2:
        effect_interpretation = "small"
    elif effect_size < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    interpretation.append(f"✓ Effect Size: Cohen's d = {effect_size:.4f}, indicating a {effect_interpretation} effect size.")
    
    # Mean and confidence interval interpretation
    if test_type == 'one_sample':
        sample_mean = results.get('Sample Mean')
        interpretation.append(f"✓ Sample Statistics: The sample mean is {sample_mean:.4f}.")
        interpretation.append(f"✓ Confidence Interval: {results.get(f'{(1-alpha)*100}% Confidence Interval')}")
        interpretation.append("   This interval represents the range where we expect the true population mean to fall.")
    else:
        mean_diff = results.get('Mean Difference')
        group1_mean = results.get('Group 1 Mean')
        group2_mean = results.get('Group 2 Mean')
        interpretation.append(f"✓ Group Comparison:")
        interpretation.append(f"   - Group 1 Mean: {group1_mean:.4f}")
        interpretation.append(f"   - Group 2 Mean: {group2_mean:.4f}")
        interpretation.append(f"   - Mean Difference: {mean_diff:.4f}")
        interpretation.append(f"✓ Confidence Interval: {results.get(f'{(1-alpha)*100}% CI of difference')}")
        interpretation.append("   This interval represents the range where we expect the true difference between population means to fall.")
    
    return interpretation

# Modify the display_statistics function
def calculate_anova(group1, group2):
    """Calculate ANOVA table for two groups."""
    # Total number of observations
    n_total = len(group1) + len(group2)
    
    # Calculate means
    grand_mean = np.mean(np.concatenate([group1, group2]))
    group1_mean = np.mean(group1)
    group2_mean = np.mean(group2)
    
    # Calculate Sum of Squares
    ss_between = (len(group1) * (group1_mean - grand_mean)**2 + 
                 len(group2) * (group2_mean - grand_mean)**2)
    
    ss_within = (np.sum((group1 - group1_mean)**2) + 
                np.sum((group2 - group2_mean)**2))
    
    ss_total = ss_between + ss_within
    
    # Degrees of freedom
    df_between = 1  # number of groups - 1
    df_within = n_total - 2  # n_total - number of groups
    df_total = n_total - 1
    
    # Mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    # F-statistic
    f_stat = ms_between / ms_within
    
    # p-value
    p_value = stats.f.sf(f_stat, df_between, df_within)
    
    # Create ANOVA table
    anova_table = pd.DataFrame({
        'Source': ['Between Groups', 'Within Groups', 'Total'],
        'SS': [ss_between, ss_within, ss_total],
        'df': [df_between, df_within, df_total],
        'MS': [ms_between, ms_within, ''],
        'F': [f_stat, '', ''],
        'p-value': [p_value, '', '']
    })
    
    return anova_table

def display_anova_table(anova_table):
    """Display formatted ANOVA table."""
    st.write("### ANOVA Table")
    
    # Format the numeric columns
    formatted_table = anova_table.copy()
    formatted_table['SS'] = formatted_table['SS'].apply(lambda x: f"{float(x):.4f}" if x != '' else '')
    formatted_table['MS'] = formatted_table['MS'].apply(lambda x: f"{float(x):.4f}" if x != '' else '')
    formatted_table['F'] = formatted_table['F'].apply(lambda x: f"{float(x):.4f}" if x != '' else '')
    formatted_table['p-value'] = formatted_table['p-value'].apply(lambda x: f"{float(x):.4f}" if x != '' else '')
    
    st.table(formatted_table)
    
    # Add ANOVA interpretation
    f_stat = float(anova_table.loc[0, 'F'])
    p_value = float(anova_table.loc[0, 'p-value'])
    st.write("#### ANOVA Interpretation")
    st.write(f"- F-statistic: {f_stat:.4f}")
    st.write(f"- p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.success("The ANOVA test shows a statistically significant difference between the groups (p < 0.05).")
        st.write("This indicates that there is strong evidence to reject the null hypothesis that all group means are equal.")
    else:
        st.warning("The ANOVA test does not show a statistically significant difference between the groups (p ≥ 0.05).")
        st.write("This indicates that there is insufficient evidence to reject the null hypothesis that all group means are equal.")

def display_statistics(results, alpha, test_type='one_sample'):
    """Display statistical results and interpretation."""
    st.write('### Results')
    for key, value in results.items():
        if isinstance(value, float):
            st.write(f'{key}: {value:.4f}')
        else:
            st.write(f'{key}: {value}')
    
    if results.get('P-value', 1) < alpha:
        st.success(f'Significant at α = {alpha}')
    else:
        st.warning(f'Not significant at α = {alpha}')
    
    # Add detailed interpretation
    st.write('### Detailed Interpretation')
    interpretations = interpret_results(results, alpha, test_type)
    for interp in interpretations:
        st.write(interp)

def calculate_descriptive_stats(data):
    """Calculate descriptive statistics for a dataset."""
    stats_dict = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Std Dev': np.std(data, ddof=1),
        'Min': np.min(data),
        'Max': np.max(data),
        'Q1': np.percentile(data, 25),
        'Q3': np.percentile(data, 75),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data)
    }
    return stats_dict

def display_descriptive_stats(data, group_name="Sample"):
    """Display descriptive statistics in a formatted way."""
    stats = calculate_descriptive_stats(data)
    
    # Display statistics in a linear layout
    st.write("Central Tendency & Dispersion:")
    st.write(f"- Mean: {stats['Mean']:.4f}")
    st.write(f"- Median: {stats['Median']:.4f}")
    st.write(f"- Standard Deviation: {stats['Std Dev']:.4f}")
    st.write(f"- Minimum: {stats['Min']:.4f}")
    st.write(f"- Maximum: {stats['Max']:.4f}")

    st.write("\nDistribution Shape:")
    st.write(f"- Q1 (25th percentile): {stats['Q1']:.4f}")
    st.write(f"- Q3 (75th percentile): {stats['Q3']:.4f}")
    st.write(f"- Skewness: {stats['Skewness']:.4f}")
    st.write(f"- Kurtosis: {stats['Kurtosis']:.4f}")
    
    # Interpret skewness and kurtosis
    st.write("### Distribution Interpretation")
    # Skewness interpretation
    if abs(stats['Skewness']) < 0.5:
        st.write("📊 The distribution is approximately symmetric")
    elif stats['Skewness'] > 0:
        st.write("📈 The distribution is positively skewed (right-tailed)")
    else:
        st.write("📉 The distribution is negatively skewed (left-tailed)")
    
    # Kurtosis interpretation
    if abs(stats['Kurtosis']) < 0.5:
        st.write("🎯 The distribution has normal tail thickness")
    elif stats['Kurtosis'] > 0:
        st.write("📌 The distribution is leptokurtic (heavy-tailed)")
    else:
        st.write("🔄 The distribution is platykurtic (light-tailed)")

def create_enhanced_visualization(data1, data2=None, test_type='one_sample'):
    """Create enhanced visualizations for the data."""
    # Input validation
    if data1 is None or len(data1) == 0:
        raise ValueError("First dataset is empty or None")
    if test_type not in ['one_sample', 'independent']:
        raise ValueError("Invalid test type")
    if test_type == 'independent' and (data2 is None or len(data2) == 0):
        raise ValueError("Second dataset is required for independent t-test")
    
    try:
        # Set style for better visualization with fallback options
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                # If both fail, continue with default style
                pass
        
        if test_type == 'one_sample':
            # Create figure with subplots
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 2)
            
            try:
                # Histogram with KDE
                ax1 = fig.add_subplot(gs[0, :])
                sns.histplot(data1, kde=True, ax=ax1)
                ax1.axvline(np.mean(data1), color='red', linestyle='--', label='Sample Mean')
                ax1.axvline(0, color='green', linestyle='--', label='Hypothesized Mean')
                ax1.legend()
                ax1.set_title('Distribution with Density Estimate')
                
                # Q-Q plot
                ax2 = fig.add_subplot(gs[1, 0])
                # Create Q-Q plot manually to ensure proper axis handling
                res = stats.probplot(data1, dist="norm", fit=True)
                ax2.scatter(res[0][0], res[0][1], color='blue', alpha=0.6)
                ax2.plot(res[0][0], res[1][0] * res[0][0] + res[1][1], color='red', linestyle='--')
                ax2.set_title('Q-Q Plot')
                ax2.set_xlabel('Theoretical Quantiles')
                ax2.set_ylabel('Sample Quantiles')
                
                # Box plot
                ax3 = fig.add_subplot(gs[1, 1])
                sns.boxplot(y=data1, ax=ax3)
                ax3.set_title('Box Plot')
                
            except Exception as e:
                plt.close(fig)
                raise ValueError(f"Error creating one-sample visualizations: {str(e)}")
        
        else:  # Independent T-Test
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 3)
            
            try:
                # Safely prepare data for visualization
                data1_clean = np.array(data1)
                data2_clean = np.array(data2)
                
                if len(data1_clean.shape) != 1 or len(data2_clean.shape) != 1:
                    raise ValueError("Data must be 1-dimensional")
                
                # Ensure data is properly aligned
                min_length = min(len(data1_clean), len(data2_clean))
                if min_length < 2:
                    raise ValueError("Each group must have at least 2 observations")
                    
                # Combined violin and box plot
                ax1 = fig.add_subplot(gs[0, :2])
                data = pd.DataFrame({
                    'Group 1': data1_clean[:min_length],
                    'Group 2': data2_clean[:min_length]
                }).melt()
                sns.violinplot(x='variable', y='value', data=data, ax=ax1, inner='box')
                ax1.set_title('Distribution Comparison')
                
                # Histograms
                ax2 = fig.add_subplot(gs[1, 0])
                sns.histplot(data1, kde=True, ax=ax2, color='blue', alpha=0.6)
                ax2.set_title('Group 1 Distribution')
                
                ax3 = fig.add_subplot(gs[1, 1])
                sns.histplot(data2, kde=True, ax=ax3, color='orange', alpha=0.6)
                ax3.set_title('Group 2 Distribution')
                
                # Q-Q plot of differences
                ax4 = fig.add_subplot(gs[1, 2])
                try:
                    # Calculate differences using aligned data
                    d1 = data1[:min_length]
                    d2 = data2[:min_length]
                    
                    # Center each group by subtracting its own mean
                    d1_centered = d1 - np.mean(d1)
                    d2_centered = d2 - np.mean(d2)
                    
                    # Calculate differences
                    diff = d1_centered - d2_centered
                    
                    # Create Q-Q plot manually
                    res = stats.probplot(diff, dist="norm", fit=True)
                    probplot_x = res[0][0]
                    probplot_y = res[0][1]
                    slope, intercept = res[1][0], res[1][1]  # Correctly unpack slope and intercept
                    
                    # Plot points and line
                    ax4.scatter(probplot_x, probplot_y, color='blue', alpha=0.6, label='Data')
                    line_y = slope * probplot_x + intercept
                    ax4.plot(probplot_x, line_y, color='red', linestyle='--', label='Theoretical')
                    
                    # Add labels and title
                    ax4.set_title('Q-Q Plot of Differences')
                    ax4.set_xlabel('Theoretical Quantiles')
                    ax4.set_ylabel('Sample Quantiles')
                    ax4.legend()
                    
                except Exception as e:
                    ax4.text(0.5, 0.5, 'Error creating Q-Q plot\nPlease check data compatibility', 
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax4.transAxes)
                    print(f"Q-Q plot error: {str(e)}")
                
            except Exception as e:
                plt.close(fig)
                raise ValueError(f"Error creating independent t-test visualizations: {str(e)}")
        
        # Adjust layout and return
        plt.tight_layout()
        return fig
        
    except Exception as e:
        plt.close('all')  # Clean up any open figures
        raise ValueError(f"Error in visualization creation: {str(e)}")
    
    finally:
        # Clean up any remaining figures except the one we want to return
        for i in plt.get_fignums():
            if plt.figure(i) != fig:
                plt.close(i)

def create_pdf_report(test_type, results, alpha, data1, data2=None):
    """Create a PDF report of the analysis results."""
    try:
        # Input validation with detailed error messages
        if data1 is None:
            raise ValueError("First dataset is None")
        if len(data1) == 0:
            raise ValueError("First dataset is empty")
        if not isinstance(data1, (list, np.ndarray, pd.Series)):
            raise ValueError("First dataset must be a list, numpy array, or pandas Series")
            
        if test_type == 'Independent T-Test':
            if data2 is None:
                raise ValueError("Second dataset is None for Independent T-Test")
            if len(data2) == 0:
                raise ValueError("Second dataset is empty for Independent T-Test")
            if not isinstance(data2, (list, np.ndarray, pd.Series)):
                raise ValueError("Second dataset must be a list, numpy array, or pandas Series")
                
        # Convert data to numpy arrays for consistent handling
        data1 = np.array(data1)
        if data2 is not None:
            data2 = np.array(data2)
        
        # Create PDF object with UTF-8 encoding
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Add header with default font
        pdf.set_font('Helvetica', 'B', 20)
        pdf.cell(0, 15, 'T-Test Analysis Report', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.line(10, 30, 200, 30)  # Add a horizontal line
        pdf.set_y(pdf.get_y() + 5)  # Replace ln(5)
        
        # Test Type
        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 10, f'Test Type: {test_type}', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Results
        if not results:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 10, 'No statistical results available', 0, 1)
            return pdf
            
        # Statistical Results
        pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, 'Statistical Results:', 0, 1)
        pdf.set_font('Helvetica', '', 12)
        
        for key, value in results.items():
            # Replace special characters
            key = key.replace('α', 'alpha')
            if isinstance(value, float):
                pdf.cell(0, 10, f'{key}: {value:.4f}', 0, 1)
            else:
                pdf.cell(0, 10, f'{key}: {value}', 0, 1)
        
        # Significance
        pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 12)
        if results.get('P-value', 1) < alpha:
            pdf.cell(0, 10, f'Result: Significant at alpha = {alpha}', 0, 1)
        else:
            pdf.cell(0, 10, f'Result: Not significant at alpha = {alpha}', 0, 1)
        
        # Add descriptive statistics
        pdf.ln(10)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, 'Descriptive Statistics:', 0, 1)
        pdf.set_font('Helvetica', '', 12)
        
        try:
            stats1 = calculate_descriptive_stats(data1)
            # Group 1/Sample Statistics
            header = "Sample Statistics:" if test_type == "One Sample T-Test" else "Group 1 Statistics:"
            pdf.cell(0, 10, header, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 10, f'Mean: {stats1["Mean"]:.4f}', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 10, f'Median: {stats1["Median"]:.4f}', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 10, f'Standard Deviation: {stats1["Std Dev"]:.4f}', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            if data2 is not None:
                pdf.set_y(pdf.get_y() + 5)  # Replace ln(5)
                pdf.set_font('Helvetica', 'B', 12)
                pdf.cell(0, 10, 'Group 2 Statistics:', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font('Helvetica', '', 12)
                stats2 = calculate_descriptive_stats(data2)
                pdf.cell(0, 10, f'Mean: {stats2["Mean"]:.4f}', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.cell(0, 10, f'Median: {stats2["Median"]:.4f}', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.cell(0, 10, f'Standard Deviation: {stats2["Std Dev"]:.4f}', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                
        except Exception as e:
            pdf.cell(0, 10, f'Error calculating descriptive statistics: {str(e)}', 0, 1)
        
        # Save figures to the PDF
        pdf.add_page()  # Start visualizations on a new page
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, 'Visualizations:', 0, 1)
        pdf.ln(5)
        
        # Create a new figure for the PDF
        try:
            # Convert test type to the format expected by visualization function
            viz_test_type = 'one_sample' if test_type == 'One Sample T-Test' else 'independent'
            fig = create_enhanced_visualization(data1, data2, test_type=viz_test_type)
            if fig is None:
                raise ValueError("Failed to create visualization")
                
            # Save the figure to a temporary file with high quality
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                # Save with high quality settings
                fig.savefig(tmp.name, format='png', bbox_inches='tight', dpi=300, 
                          facecolor='white', edgecolor='none', pad_inches=0.1)
                plt.close(fig)  # Close the figure to free memory
                
                # Calculate dimensions to maintain aspect ratio
                img_width = pdf.w - 20  # 10mm margins on each side
                img_x = 10  # Left margin
                
                # Add the image with proper spacing
                pdf.image(tmp.name, x=img_x, w=img_width)
                os.unlink(tmp.name)  # Delete the temporary file
        except Exception as e:
            pdf.cell(0, 10, f'Error adding visualization: {str(e)}', 0, 1)
            plt.close('all')  # Ensure all figures are cleaned up in case of error
        
        return pdf
        
    except Exception as e:
        # Clean up any matplotlib figures if an error occurs
        plt.close('all')
        raise ValueError(f"Error generating PDF report: {str(e)}")
        
    finally:
        # Always clean up matplotlib figures
        plt.close('all')
def get_pdf_download_link(pdf, filename="t_test_analysis.pdf"):
    """Generate a download link for the PDF file."""
    try:
        with st.spinner('Generating PDF report...'):
            if pdf is None:
                st.error("No PDF report available - data may be empty or invalid")
                return ""
            
            try:
                # Get PDF bytes directly without additional encoding
                pdf_bytes = pdf.output()  # This already returns bytes
                b64 = base64.b64encode(pdf_bytes).decode()
                
                if not b64:
                    st.warning("Generated PDF is empty")
                    return ""
                
                # Create download link with proper MIME type and encoding
                href = f'data:application/pdf;base64,{b64}'
                return f'<div style="text-align: center;"><a href="{href}" download="{filename}" class="pdf-download-link">📊 Download Analysis Report (PDF)</a></div>'
                
            except Exception as encoding_error:
                st.error(f"Error encoding PDF: {str(encoding_error)}")
                return ""
                
    except Exception as e:
        st.error(f"Error generating PDF download link: {str(e)}")
        return ""

# Sidebar configuration
with st.sidebar:
    st.header('Configuration')
    test_type = st.selectbox(
        'Select T-Test Type',
        ['One Sample T-Test', 'Independent T-Test']
    )
    alpha = st.slider('Significance Level (α)', 0.01, 0.10, 0.05, 0.01)
    st.write(f'Current α = {alpha}')

# Main application logic
if test_type == 'One Sample T-Test':
    st.subheader('One Sample T-Test')
    
    uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        df = read_file_data(uploaded_file)
        if df is not None:
            selected_column = st.selectbox("Select column for analysis", df.columns)
            data = df[selected_column].dropna().values
            st.write(f"Preview of {selected_column}:", data[:5])
            
            hypothesized_mean = st.number_input('Enter hypothesized mean:', value=0.0)
            
            if st.button('Calculate'):
                try:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_1samp(data, hypothesized_mean)
                    ci = stats.t.interval(1-alpha, len(data)-1, loc=np.mean(data), 
                                       scale=stats.sem(data))
                    effect_size = calculate_effect_size(data, test_type='one_sample')
                    
                    results = {
                        'Sample Mean': np.mean(data),
                        'Sample Size': len(data),
                        'T-statistic': t_stat,
                        'P-value': p_value,
                        'Effect Size (Cohen\'s d)': effect_size,
                        f'{(1-alpha)*100}% Confidence Interval': f'[{ci[0]:.4f}, {ci[1]:.4f}]'
                    }
                    
                    # Add descriptive statistics
                    display_descriptive_stats(data, "Sample Data")
                    
                    # Display test results
                    display_statistics(results, alpha, 'one_sample')
                    
                    # Display visualization
                    st.write('### Enhanced Visualization')
                    try:
                        # Convert test type to the format expected by visualization function
                        viz_test_type = 'one_sample' if test_type == 'One Sample T-Test' else 'independent'
                        fig = create_enhanced_visualization(data, test_type=viz_test_type)
                        if fig is not None:
                            st.pyplot(fig)
                        else:
                            st.warning("Could not create visualization")
                    except Exception as viz_error:
                        st.error(f"Error creating visualization: {str(viz_error)}")
                        plt.close('all')  # Clean up any open figures

                    # Generate and display download button
                    try:
                        pdf = create_pdf_report(test_type, results, alpha, data)
                        download_link = get_pdf_download_link(pdf, "one_sample_t_test_report.pdf")
                        if download_link:
                            st.markdown(download_link, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Could not generate PDF report: {str(e)}")
                    
                except Exception as e:
                    st.error(f'Error: {str(e)}. Please check your input data.')

else:  # Independent T-Test
    st.subheader('Independent T-Test')
    
    uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", 
                                   type=['csv', 'xlsx', 'xls'], 
                                   key="independent_test")
    
    if uploaded_file is not None:
        df = read_file_data(uploaded_file)
        if df is not None:
            st.markdown("""
            ### Column Selection Guide
            - **Group 1**: Select the column containing your first group's measurements (e.g., control group, pre-treatment)
            - **Group 2**: Select the column containing your second group's measurements (e.g., treatment group, post-treatment)
            
            Both columns should contain numerical data with similar scales of measurement.
            """)
            
            group1_col = st.selectbox(
                "Choose column for Group 1 (e.g., Control Group)", 
                df.columns,
                help="Select the column containing measurements for your first group (e.g., control group, pre-treatment, method A)"
            )
            group2_col = st.selectbox(
                "Choose column for Group 2 (e.g., Treatment Group)", 
                df.columns,
                help="Select the column containing measurements for your second group (e.g., treatment group, post-treatment, method B)"
            )
            
            group1 = df[group1_col].dropna().values
            group2 = df[group2_col].dropna().values
            
            if len(group1) < 2 or len(group2) < 2:
                st.error("Each group must have at least 2 observations")
            else:
                st.write(f"Preview of {group1_col}:", group1[:5])
                st.write(f"Preview of {group2_col}:", group2[:5])
                
                if st.button('Calculate'):
                    try:
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(group1, group2)
                        effect_size = calculate_effect_size(group1, group2, test_type='independent')
                        diff_ci = stats.t.interval(1-alpha, len(group1)+len(group2)-2,
                                                 loc=np.mean(group1)-np.mean(group2),
                                                 scale=np.sqrt(np.var(group1)/len(group1) + 
                                                             np.var(group2)/len(group2)))
                        
                        results = {
                            'Group 1 Mean': np.mean(group1),
                            'Group 2 Mean': np.mean(group2),
                            'Mean Difference': np.mean(group1) - np.mean(group2),
                            'Group 1 Size': len(group1),
                            'Group 2 Size': len(group2),
                            'T-statistic': t_stat,
                            'P-value': p_value,
                            'Effect Size (Cohen\'s d)': effect_size,
                            f'{(1-alpha)*100}% CI of difference': f'[{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]'
                        }
                        
                        # Display descriptive statistics in a linear layout
                        st.write("### Group Statistics")
                        st.write("#### Group 1")
                        display_descriptive_stats(group1, "Group 1")
                        st.write("#### Group 2")
                        display_descriptive_stats(group2, "Group 2")
                        
                        # Display test results
                        display_statistics(results, alpha, 'independent')
                        
                        # Create and display visualizations
                        st.write('### Enhanced Visualization')
                        fig = create_enhanced_visualization(group1, group2, test_type='independent')
                        st.pyplot(fig)
                        
                        # Add ANOVA analysis
                        anova_table = calculate_anova(group1, group2)
                        display_anova_table(anova_table)
                        
                        # Generate and display download button
                        pdf = create_pdf_report(test_type, results, alpha, group1, group2)
                        download_link = get_pdf_download_link(pdf, "independent_t_test_report.pdf")
                        if download_link:
                            st.markdown(download_link, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f'Error: {str(e)}. Please check your input data.')

# Add general interpretations
st.markdown("""
### General Guidelines for Interpretation:
- Statistical Significance (P-value):
  - P-value < α: Strong evidence against the null hypothesis
  - P-value ≥ α: Insufficient evidence against the null hypothesis
  
- Effect Size (Cohen's d):
  - Small effect: |d| < 0.2
  - Medium effect: 0.2 ≤ |d| < 0.8
  - Large effect: |d| ≥ 0.8
  
- Confidence Interval:
  - Represents the range of plausible values for the true population parameter
  - Narrower intervals indicate more precise estimates
  - If the interval includes zero, it aligns with non-significant results
""")

# Add footer with creator information
st.markdown("---")  # Horizontal line to separate content from footer
st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        <p>Created by <a href="https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/" target="_blank">Galuh Adi Insani</a> with ❤️</p>
    </div>
    """,
    unsafe_allow_html=True
)
