#!/usr/bin/env python3
"""
Data Analysis Script for Excel Files
Analyzes Media & Research Articles data.xlsx and Twitter Posts Data.xlsx
"""

import pandas as pd
import os
from pathlib import Path

def analyze_excel_file(file_path, file_name):
    """Analyze a single Excel file and return comprehensive information"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {file_name}")
    print(f"{'='*60}")
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Basic information
        print(f"\nBASIC INFORMATION:")
        print(f"   - File size: {os.path.getsize(file_path) / 1024:.1f} KB")
        print(f"   - Number of rows: {len(df)}")
        print(f"   - Number of columns: {len(df.columns)}")
        
        # Column information
        print(f"\nCOLUMNS ({len(df.columns)} total):")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            null_count = len(df) - non_null
            print(f"   {i:2d}. {col}")
            print(f"       Type: {dtype}, Non-null: {non_null}, Null: {null_count}")
        
        # Data types summary
        print(f"\nDATA TYPES SUMMARY:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   - {dtype}: {count} columns")
        
        # Missing data analysis
        print(f"\nMISSING DATA ANALYSIS:")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"   Total missing values: {missing_data.sum()}")
            for col, missing in missing_data.items():
                if missing > 0:
                    percentage = (missing / len(df)) * 100
                    print(f"   - {col}: {missing} missing ({percentage:.1f}%)")
        else:
            print("   No missing data found!")
        
        # Sample data preview
        print(f"\nFIRST 3 ROWS PREVIEW:")
        print(df.head(3).to_string(max_cols=None, max_colwidth=50))
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            print(f"\nNUMERIC COLUMNS ANALYSIS:")
            print(df[numeric_cols].describe())
        
        # Text columns analysis
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            print(f"\nTEXT COLUMNS ANALYSIS:")
            for col in text_cols[:3]:  # Show analysis for first 3 text columns
                unique_count = df[col].nunique()
                most_common = df[col].value_counts().head(3)
                print(f"\n   Column: {col}")
                print(f"   - Unique values: {unique_count}")
                print(f"   - Most common values:")
                for value, count in most_common.items():
                    print(f"     - '{str(value)[:50]}...': {count} times")
        
        return df
        
    except Exception as e:
        print(f"Error reading {file_name}: {str(e)}")
        return None

def main():
    """Main analysis function"""
    print("EXCEL FILES DATA ANALYSIS")
    print("=" * 60)
    
    # Define data directory
    data_dir = Path("data")
    
    # File paths
    files_to_analyze = [
        ("Media & Research Articles data.xlsx", "Media & Research Articles Data"),
        ("Twitter Posts Data.xlsx", "Twitter Posts Data")
    ]
    
    analysis_results = {}
    
    # Analyze each file
    for filename, display_name in files_to_analyze:
        file_path = data_dir / filename
        if file_path.exists():
            df = analyze_excel_file(file_path, display_name)
            if df is not None:
                analysis_results[display_name] = df
        else:
            print(f"File not found: {file_path}")
    
    # Summary comparison
    if len(analysis_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        
        for name, df in analysis_results.items():
            print(f"\n{name}:")
            print(f"   - Rows: {len(df)}")
            print(f"   - Columns: {len(df.columns)}")
            print(f"   - Data types: {df.dtypes.nunique()} different types")
            print(f"   - Missing values: {df.isnull().sum().sum()}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
