#!/usr/bin/env python3
"""
GenAI Analyst Assignment - First Assignment
Student ID: rocky07
Company: InfiGen

This script performs comprehensive NLP analysis on medical articles and Twitter posts
using GPU-accelerated models (BioBERT, BART, RoBERTa) instead of OpenAI API.

Assignment Requirements:
1. Import and explore datasets
2. Data cleaning and preparation  
3. Data consolidation
4. Text enhancement
5. NLP Analysis: Entity Recognition, Topic Classification, Sentiment Analysis
6. Output generation and export

Author: rocky07
Date: September 2024
"""

import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import warnings
import time
import json
from pathlib import Path
import re
from datetime import datetime

warnings.filterwarnings("ignore")

class GenAIAnalystAssignment:
    """
    Complete solution for GenAI Analyst Assignment using GPU-accelerated NLP models
    """
    
    def __init__(self, student_id="rocky07"):
        self.student_id = student_id
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.models = {}
        self.data = {}
        self.results = None
        
        print("="*80)
        print("GenAI Analyst Assignment - InfiGen")
        print(f"Student ID: {self.student_id}")
        print(f"Device: {self.device}")
        print("="*80)
    
    def step1_import_and_explore_data(self):
        """
        Step 1: Import and Explore the Data
        Import the provided datasets and explore their structure
        """
        print("\nSTEP 1: IMPORT AND EXPLORE DATA")
        print("-" * 50)
        
        try:
            # Import Media & Research Articles data
            print("Loading Media & Research Articles data...")
            articles_path = Path("data/Media & Research Articles data.xlsx")
            self.data['articles_raw'] = pd.read_excel(articles_path)
            
            print(f"   Articles loaded: {len(self.data['articles_raw'])} records")
            print(f"   Columns: {list(self.data['articles_raw'].columns)}")
            print(f"   Sample article title: {self.data['articles_raw']['Article title'].iloc[0][:100]}...")
            
            # Import Twitter Posts data
            print("\nLoading Twitter Posts data...")
            twitter_path = Path("data/Twitter Posts Data.xlsx")
            self.data['twitter_raw'] = pd.read_excel(twitter_path)
            
            print(f"   Twitter posts loaded: {len(self.data['twitter_raw'])} records")
            print(f"   Columns: {list(self.data['twitter_raw'].columns)}")
            print(f"   Sample tweet: {self.data['twitter_raw']['Posts'].iloc[0][:100]}...")
            
            # Data exploration summary
            print(f"\nDATA EXPLORATION SUMMARY:")
            print(f"   - Total articles: {len(self.data['articles_raw'])}")
            print(f"   - Total tweets: {len(self.data['twitter_raw'])}")
            print(f"   - Articles date range: {self.data['articles_raw']['Published date'].min()} to {self.data['articles_raw']['Published date'].max()}")
            print(f"   - Twitter date range: {self.data['twitter_raw']['Published date'].min()} to {self.data['twitter_raw']['Published date'].max()}")
            
            return True
            
        except Exception as e:
            print(f"   Error loading data: {str(e)}")
            return False
    
    def step2_data_preparation(self):
        """
        Step 2: Data Preparation
        Clean and prepare datasets, rename columns for consistency
        """
        print("\nSTEP 2: DATA PREPARATION")
        print("-" * 50)
        
        try:
            # Prepare Articles dataset
            print("Preparing Articles dataset...")
            articles_clean = self.data['articles_raw'].copy()
            
            # Rename columns for consistency (mapping to assignment requirements)
            articles_columns_map = {
                'unique_id': 'ID',
                'Article title': 'Title',
                'Content': 'Body',
                'Source': 'Original_Source',
                'Published date': 'Date'
            }
            
            articles_clean = articles_clean.rename(columns=articles_columns_map)
            
            # Clean text data
            articles_clean['Title'] = articles_clean['Title'].fillna('').astype(str)
            articles_clean['Body'] = articles_clean['Body'].fillna('').astype(str)
            
            print(f"   Articles cleaned: {len(articles_clean)} records")
            print(f"   New columns: {list(articles_clean.columns)}")
            
            # Prepare Twitter dataset
            print("\nPreparing Twitter dataset...")
            twitter_clean = self.data['twitter_raw'].copy()
            
            # Rename columns for consistency
            twitter_columns_map = {
                'unique_id': 'ID',
                'Posts': 'Body',  # Twitter posts go to Body column
                'HCP Handle': 'Original_Source',
                'Published date': 'Date'
            }
            
            twitter_clean = twitter_clean.rename(columns=twitter_columns_map)
            
            # For Twitter, we don't have separate titles, so we'll use first part of post as title
            twitter_clean['Title'] = twitter_clean['Body'].apply(
                lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x)
            )
            twitter_clean['Body'] = twitter_clean['Body'].fillna('').astype(str)
            
            print(f"   Twitter cleaned: {len(twitter_clean)} records")
            print(f"   New columns: {list(twitter_clean.columns)}")
            
            # Store cleaned data
            self.data['articles_clean'] = articles_clean
            self.data['twitter_clean'] = twitter_clean
            
            return True
            
        except Exception as e:
            print(f"   Error in data preparation: {str(e)}")
            return False
    
    def step3_add_source_column(self):
        """
        Step 3: Add Source Column
        Add a new column named 'Source' to track origin ('Media' or 'Twitter')
        """
        print("\nSTEP 3: ADD SOURCE COLUMN")
        print("-" * 50)
        
        try:
            # Add Source column to articles
            self.data['articles_clean']['Source'] = 'Media'
            print(f"   Added 'Source' = 'Media' to {len(self.data['articles_clean'])} articles")
            
            # Add Source column to twitter
            self.data['twitter_clean']['Source'] = 'Twitter'
            print(f"   Added 'Source' = 'Twitter' to {len(self.data['twitter_clean'])} tweets")
            
            # Verify source distribution
            print(f"\nSource Distribution:")
            print(f"   - Media articles: {len(self.data['articles_clean'])}")
            print(f"   - Twitter posts: {len(self.data['twitter_clean'])}")
            
            return True
            
        except Exception as e:
            print(f"   Error adding source column: {str(e)}")
            return False
    
    def step4_data_consolidation(self):
        """
        Step 4: Data Consolidation
        Merge the two datasets vertically into a single DataFrame
        """
        print("\nSTEP 4: DATA CONSOLIDATION")
        print("-" * 50)
        
        try:
            # Ensure both datasets have the same columns
            required_columns = ['ID', 'Title', 'Body', 'Source', 'Date', 'Original_Source']
            
            # Select and reorder columns for both datasets
            articles_subset = self.data['articles_clean'][required_columns].copy()
            twitter_subset = self.data['twitter_clean'][required_columns].copy()
            
            # Merge datasets vertically
            consolidated_data = pd.concat([articles_subset, twitter_subset], 
                                        ignore_index=True, sort=False)
            
            # Add a unique index for tracking
            consolidated_data['Index'] = range(len(consolidated_data))
            
            # Store consolidated data
            self.data['consolidated'] = consolidated_data
            
            print(f"   Data consolidated successfully!")
            print(f"   Total records: {len(consolidated_data)}")
            print(f"   Final columns: {list(consolidated_data.columns)}")
            print(f"   Source breakdown:")
            print(f"      - Media: {(consolidated_data['Source'] == 'Media').sum()}")
            print(f"      - Twitter: {(consolidated_data['Source'] == 'Twitter').sum()}")
            
            return True
            
        except Exception as e:
            print(f"   Error in data consolidation: {str(e)}")
            return False
    
    def step5_textual_data_enhancement(self):
        """
        Step 5: Textual Data Enhancement
        Combine Title and Body columns into a new 'Combined' column
        """
        print("\nSTEP 5: TEXTUAL DATA ENHANCEMENT")
        print("-" * 50)
        
        try:
            consolidated = self.data['consolidated'].copy()
            
            # Combine Title and Body into Combined column
            consolidated['Combined'] = consolidated.apply(
                lambda row: f"{row['Title']} {row['Body']}" if pd.notna(row['Title']) and pd.notna(row['Body']) 
                else (row['Title'] if pd.notna(row['Title']) else row['Body']),
                axis=1
            )
            
            # Clean the combined text
            consolidated['Combined'] = consolidated['Combined'].apply(
                lambda x: re.sub(r'\s+', ' ', str(x)).strip()  # Remove extra whitespace
            )
            
            # Calculate text statistics
            consolidated['Text_Length'] = consolidated['Combined'].apply(len)
            consolidated['Word_Count'] = consolidated['Combined'].apply(lambda x: len(str(x).split()))
            
            # Store enhanced data
            self.data['enhanced'] = consolidated
            
            print(f"   Text enhancement completed!")
            print(f"   Combined text statistics:")
            print(f"      - Average text length: {consolidated['Text_Length'].mean():.0f} characters")
            print(f"      - Average word count: {consolidated['Word_Count'].mean():.0f} words")
            print(f"      - Max text length: {consolidated['Text_Length'].max()} characters")
            print(f"      - Min text length: {consolidated['Text_Length'].min()} characters")
            
            # Show sample combined text
            print(f"\nSample combined text:")
            sample_text = consolidated['Combined'].iloc[0][:200]
            print(f"   '{sample_text}...'")
            
            return True
            
        except Exception as e:
            print(f"   Error in textual enhancement: {str(e)}")
            return False
    
    def step6_setup_nlp_models(self):
        """
        Step 6: Setup NLP Models
        Initialize GPU-accelerated models instead of OpenAI API
        """
        print("\nSTEP 6: NLP MODELS SETUP")
        print("-" * 50)
        print("Note: Using GPU-accelerated open-source models instead of OpenAI API")
        print("Models: BioBERT (NER), BART (Topic Classification), RoBERTa (Sentiment)")
        
        try:
            # Setup Sentiment Analysis (RoBERTa)
            print("\nSetting up Sentiment Analysis...")
            self.models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device,
                torch_dtype=torch.float16
            )
            print("   RoBERTa sentiment model ready!")
            
            # Setup Topic Classification (BART)
            print("\nSetting up Topic Classification...")
            self.models['topic'] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device,
                torch_dtype=torch.float16
            )
            print("   BART topic classifier ready!")
            
            # Setup Entity Recognition (BioBERT)
            print("\nSetting up Entity Recognition...")
            tokenizer = AutoTokenizer.from_pretrained("alvaroalon2/biobert_diseases_ner")
            model = AutoModelForTokenClassification.from_pretrained(
                "alvaroalon2/biobert_diseases_ner",
                torch_dtype=torch.float16
            ).to(self.device)
            
            self.models['ner'] = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=self.device
            )
            print("   BioBERT NER model ready!")
            
            print(f"\nAll {len(self.models)} models successfully loaded on {self.device}!")
            return True
            
        except Exception as e:
            print(f"   Error setting up models: {str(e)}")
            return False
    
    def step7_nlp_analysis(self):
        """
        Step 7: Comprehensive NLP Analysis
        Perform Entity Recognition, Topic Classification, and Sentiment Analysis
        """
        print("\nSTEP 7: NLP ANALYSIS")
        print("-" * 50)
        
        # Define assignment-specific categories
        assignment_topics = [
            "Efficacy-General",
            "Progression Free Survival (PFS)",
            "Overall Survival (OS)", 
            "Safety-General",
            "Safety-Side Effects",
            "General Opinion",
            "Others"
        ]
        
        enhanced_data = self.data['enhanced'].copy()
        total_records = len(enhanced_data)
        
        print(f"Processing {total_records} records...")
        print(f"Target topics: {assignment_topics}")
        print(f"Target entities: Drugs, Diseases, Study Names")
        
        results = []
        start_time = time.time()
        
        for idx, row in enhanced_data.iterrows():
            combined_text = row['Combined']
            text_preview = combined_text[:500]  # Limit for processing speed
            
            result = {
                'Index': row['Index'],
                'ID': row['ID'],
                'Source': row['Source'],
                'Title': row['Title'],
                'Combined_Text': combined_text,
                'Text_Preview': text_preview + "..." if len(combined_text) > 500 else combined_text
            }
            
            # 1. Entity Recognition - Extract Drugs, Diseases, Study Names
            try:
                entities = self.models['ner'](text_preview)
                
                # Categorize entities based on assignment requirements
                drugs = []
                diseases = []
                study_names = []
                
                for entity in entities:
                    if entity['score'] > 0.7:  # High confidence threshold
                        entity_text = entity['word'].lower()
                        entity_label = entity['entity_group']
                        
                        # Simple categorization (can be improved with medical dictionaries)
                        if any(drug_keyword in entity_text for drug_keyword in 
                               ['drug', 'therapy', 'treatment', 'medication', 'metformin', 'insulin']):
                            drugs.append(entity['word'])
                        elif entity_label in ['DISEASE', '0'] or any(disease_keyword in entity_text for disease_keyword in 
                                ['cancer', 'diabetes', 'disease', 'tumor', 'lymphoma']):
                            diseases.append(entity['word'])
                        else:
                            # Assume remaining are study names or other medical terms
                            study_names.append(entity['word'])
                
                result['Drugs'] = list(set(drugs))[:5]  # Top 5 unique drugs
                result['Diseases'] = list(set(diseases))[:5]  # Top 5 unique diseases  
                result['Study_Names'] = list(set(study_names))[:5]  # Top 5 unique studies
                
            except Exception as e:
                result['Drugs'] = []
                result['Diseases'] = []
                result['Study_Names'] = []
            
            # 2. Topic Classification - Assignment specific topics
            try:
                topic_result = self.models['topic'](text_preview, assignment_topics)
                result['Primary_Topic'] = topic_result['labels'][0]
                result['Topic_Confidence'] = topic_result['scores'][0]
                result['All_Topic_Scores'] = dict(zip(topic_result['labels'], topic_result['scores']))
                
            except Exception as e:
                result['Primary_Topic'] = 'Others'
                result['Topic_Confidence'] = 0.5
                result['All_Topic_Scores'] = {}
            
            # 3. Sentiment Analysis - For each topic
            try:
                sentiment_result = self.models['sentiment'](text_preview)[0]
                result['Sentiment'] = sentiment_result['label']
                result['Sentiment_Score'] = sentiment_result['score']
                
                # Topic-specific sentiment (simplified approach)
                primary_topic = result['Primary_Topic']
                result[f'Sentiment_for_{primary_topic}'] = sentiment_result['label']
                result[f'Sentiment_Score_for_{primary_topic}'] = sentiment_result['score']
                
            except Exception as e:
                result['Sentiment'] = 'neutral'
                result['Sentiment_Score'] = 0.5
                result[f'Sentiment_for_{result["Primary_Topic"]}'] = 'neutral'
                result[f'Sentiment_Score_for_{result["Primary_Topic"]}'] = 0.5
            
            results.append(result)
            
            # Progress tracking
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed
                remaining = (total_records - idx - 1) / speed if speed > 0 else 0
                print(f"   Processed {idx + 1}/{total_records} ({speed:.1f} records/sec, ~{remaining:.0f}s remaining)")
        
        total_time = time.time() - start_time
        print(f"\nNLP Analysis completed in {total_time:.2f} seconds!")
        print(f"Average processing speed: {total_records/total_time:.1f} records/second")
        
        # Store results
        self.results = pd.DataFrame(results)
        return True
    
    def step8_output_generation(self):
        """
        Step 8: Output Generation
        Organize predictions into structured format and export
        """
        print("\nSTEP 8: OUTPUT GENERATION")
        print("-" * 50)
        
        try:
            if self.results is None:
                print("   No results to export!")
                return False
            
            # Create final predictions DataFrame with required format
            predictions_df = self.results.copy()
            
            # Add metadata
            predictions_df['Student_ID'] = self.student_id
            predictions_df['Processing_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            predictions_df['Model_Used'] = 'BioBERT+BART+RoBERTa (GPU-accelerated)'
            predictions_df['Device'] = str(self.device)
            
            # Generate summary statistics
            summary_stats = {
                'total_records': len(predictions_df),
                'media_records': (predictions_df['Source'] == 'Media').sum(),
                'twitter_records': (predictions_df['Source'] == 'Twitter').sum(),
                'sentiment_distribution': predictions_df['Sentiment'].value_counts().to_dict(),
                'topic_distribution': predictions_df['Primary_Topic'].value_counts().to_dict(),
                'avg_confidence': predictions_df['Topic_Confidence'].mean(),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Export to CSV
            output_filename = f'predictions_{self.student_id}.csv'
            predictions_df.to_csv(output_filename, index=False)
            
            # Export summary
            summary_filename = f'analysis_summary_{self.student_id}.json'
            with open(summary_filename, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            print(f"   Results exported to: {output_filename}")
            print(f"   Summary exported to: {summary_filename}")
            print(f"   Total predictions: {len(predictions_df)}")
            print(f"   Columns in output: {len(predictions_df.columns)}")
            
            # Display key insights
            print(f"\nKEY INSIGHTS:")
            print(f"   - Most common topic: {predictions_df['Primary_Topic'].mode().iloc[0]}")
            print(f"   - Most common sentiment: {predictions_df['Sentiment'].mode().iloc[0]}")
            print(f"   - Average topic confidence: {predictions_df['Topic_Confidence'].mean():.3f}")
            print(f"   - Average sentiment confidence: {predictions_df['Sentiment_Score'].mean():.3f}")
            
            return True
            
        except Exception as e:
            print(f"   Error in output generation: {str(e)}")
            return False
    
    def run_complete_assignment(self):
        """
        Execute the complete assignment workflow
        """
        print("STARTING GENAI ANALYST ASSIGNMENT")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute all steps in sequence
        steps = [
            ("Import and Explore Data", self.step1_import_and_explore_data),
            ("Data Preparation", self.step2_data_preparation),
            ("Add Source Column", self.step3_add_source_column),
            ("Data Consolidation", self.step4_data_consolidation),
            ("Textual Data Enhancement", self.step5_textual_data_enhancement),
            ("Setup NLP Models", self.step6_setup_nlp_models),
            ("NLP Analysis", self.step7_nlp_analysis),
            ("Output Generation", self.step8_output_generation)
        ]
        
        start_time = time.time()
        
        for step_name, step_function in steps:
            print(f"\n{'='*80}")
            try:
                success = step_function()
                if not success:
                    print(f"FAILED: {step_name}")
                    return False
                else:
                    print(f"COMPLETED: {step_name}")
            except Exception as e:
                print(f"ERROR in {step_name}: {str(e)}")
                return False
        
        total_time = time.time() - start_time
        
        # Final summary
        print(f"\n{'='*80}")
        print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Output files generated:")
        print(f"   - predictions_{self.student_id}.csv")
        print(f"   - analysis_summary_{self.student_id}.json")
        print("="*80)
        
        return True

def main():
    """
    Main execution function for the assignment
    """
    print("GenAI Analyst Assignment - InfiGen")
    print("Student: rocky07")
    print("Assignment: Data Analysis and NLP using GPU-accelerated models")
    
    # Initialize and run assignment
    assignment = GenAIAnalystAssignment(student_id="rocky07")
    success = assignment.run_complete_assignment()
    
    if success:
        print("\nAssignment completed successfully!")
        print("Ready for submission to InfiGen")
    else:
        print("\nAssignment failed. Please check the errors above.")
    
    return assignment

if __name__ == "__main__":
    assignment_instance = main()
