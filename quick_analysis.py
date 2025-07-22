#!/usr/bin/env python3
"""
Quick Data Analysis Script
=========================

This script provides a quick overview of the tax data to understand
the structure and patterns before running the full ML pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def quick_data_analysis():
    """Perform quick analysis of the tax data"""
    print("Loading data for quick analysis...")
    
    # Load sample data (first 10000 rows for speed)
    print("Loading sample data...")
    arrears_sample = pd.read_csv('arrears_query.csv', nrows=10000)
    bills_sample = pd.read_csv('bill_query.csv', nrows=10000)
    collections_sample = pd.read_csv('collection_query.csv', nrows=10000)
    
    print(f"Arrears sample: {len(arrears_sample)} rows")
    print(f"Bills sample: {len(bills_sample)} rows")
    print(f"Collections sample: {len(collections_sample)} rows")
    
    # Basic statistics
    print("\n" + "="*50)
    print("BASIC STATISTICS")
    print("="*50)
    
    print("\nArrears Data:")
    print(f"- Total arrears amount: ₹{arrears_sample['ARREARSAMOUNT'].sum():,.2f}")
    print(f"- Average arrears: ₹{arrears_sample['ARREARSAMOUNT'].mean():,.2f}")
    print(f"- Zero arrears count: {(arrears_sample['ARREARSAMOUNT'] == 0).sum()}")
    
    print("\nBills Data:")
    print(f"- Total bills amount: ₹{bills_sample['CURRENT_S1_S2'].sum():,.2f}")
    print(f"- Average bill: ₹{bills_sample['CURRENT_S1_S2'].mean():,.2f}")
    print(f"- Zero bills count: {(bills_sample['CURRENT_S1_S2'] == 0).sum()}")
    
    print("\nCollections Data:")
    print(f"- Total collections: ₹{collections_sample['COLL_TOTAL'].sum():,.2f}")
    print(f"- Current collections: ₹{collections_sample['COLL_CURRENT'].sum():,.2f}")
    print(f"- Arrears collections: ₹{collections_sample['COLL_ARREARS'].sum():,.2f}")
    
    # Tax types analysis
    print("\n" + "="*50)
    print("TAX TYPES ANALYSIS")
    print("="*50)
    
    print("\nTop tax types in arrears:")
    arrears_by_tax = arrears_sample.groupby('VAR_TAX_ENAME')['ARREARSAMOUNT'].sum().sort_values(ascending=False)
    print(arrears_by_tax.head(10))
    
    print("\nTop tax types in bills:")
    bills_by_tax = bills_sample.groupby('VAR_TAX_ENAME')['CURRENT_S1_S2'].sum().sort_values(ascending=False)
    print(bills_by_tax.head(10))
    
    print("\nTop tax types in collections:")
    collections_by_tax = collections_sample.groupby('VAR_TAX_ENAME')['COLL_TOTAL'].sum().sort_values(ascending=False)
    print(collections_by_tax.head(10))
    
    # Collection efficiency
    print("\n" + "="*50)
    print("COLLECTION EFFICIENCY")
    print("="*50)
    
    # Calculate collection rate where possible
    non_zero_collections = collections_sample[collections_sample['COLL_TOTAL'] > 0]
    if len(non_zero_collections) > 0:
        print(f"Properties with collections: {len(non_zero_collections)}")
        print(f"Average collection amount: ₹{non_zero_collections['COLL_TOTAL'].mean():,.2f}")
    
    # ULB analysis
    print("\n" + "="*50)
    print("ULB PERFORMANCE")
    print("="*50)
    
    ulb_collections = collections_sample.groupby('NUM_REC_ULBID')['COLL_TOTAL'].sum().sort_values(ascending=False)
    print("Top 10 ULBs by collections:")
    print(ulb_collections.head(10))
    
    # Create simple visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Tax type distribution in collections
    plt.subplot(2, 2, 1)
    top_taxes_collections = collections_sample.groupby('VAR_TAX_ENAME')['COLL_TOTAL'].sum().sort_values(ascending=False).head(10)
    plt.barh(range(len(top_taxes_collections)), top_taxes_collections.values)
    plt.yticks(range(len(top_taxes_collections)), top_taxes_collections.index)
    plt.title('Top 10 Tax Types by Collections')
    plt.xlabel('Collection Amount')
    
    # Plot 2: Collection amounts distribution
    plt.subplot(2, 2, 2)
    non_zero_collections = collections_sample[collections_sample['COLL_TOTAL'] > 0]['COLL_TOTAL']
    plt.hist(non_zero_collections, bins=50, edgecolor='black')
    plt.title('Distribution of Collection Amounts')
    plt.xlabel('Collection Amount')
    plt.ylabel('Frequency')
    plt.yscale('log')
    
    # Plot 3: ULB performance
    plt.subplot(2, 2, 3)
    top_ulbs = collections_sample.groupby('NUM_REC_ULBID')['COLL_TOTAL'].sum().sort_values(ascending=False).head(10)
    plt.bar(range(len(top_ulbs)), top_ulbs.values)
    plt.title('Top 10 ULBs by Collections')
    plt.xlabel('ULB Rank')
    plt.ylabel('Collection Amount')
    
    # Plot 4: Collection vs Arrears
    plt.subplot(2, 2, 4)
    sample_for_scatter = collections_sample.sample(min(1000, len(collections_sample)))
    plt.scatter(sample_for_scatter['COLL_ARREARS'], sample_for_scatter['COLL_CURRENT'], alpha=0.6)
    plt.title('Current vs Arrears Collections')
    plt.xlabel('Arrears Collections')
    plt.ylabel('Current Collections')
    
    plt.tight_layout()
    plt.savefig('quick_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nQuick analysis completed! Visualization saved as 'quick_analysis.png'")
    
    # Recommendations
    print("\n" + "="*50)
    print("RECOMMENDATIONS FOR ML MODEL")
    print("="*50)
    print("1. Focus on high-value tax categories like General Tax")
    print("2. Consider ULB-specific models due to performance variation")
    print("3. Handle zero-amount records appropriately")
    print("4. Create features based on historical collection patterns")
    print("5. Consider seasonal/temporal factors if date information available")

if __name__ == "__main__":
    quick_data_analysis()
