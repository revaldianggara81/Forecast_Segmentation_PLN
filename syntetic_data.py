import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_power_data():
    """
    Generate synthetic hourly household power consumption data 
    with demographic information including income and region data for Indonesia
    """
    
    # Indonesian regions with approximate coordinates
    indonesian_regions = {
        # Jawa & Jakarta
        'Jakarta': {
            'lat_range': (-6.25, -6.05), 
            'lon_range': (106.7, 106.95)
        },
        'Jawa Barat': {
            'lat_range': (-7.0, -5.9), 
            'lon_range': (106.3, 108.3)
        },
        'Jawa Tengah': {
            'lat_range': (-7.7, -6.5), 
            'lon_range': (109.0, 111.5)
        },
        'Jawa Timur': {
            'lat_range': (-8.3, -7.0), 
            'lon_range': (111.0, 114.3)
        },

        # Sumatera
        'Sumatera Utara': {
            'lat_range': (2.0, 3.8), 
            'lon_range': (97.0, 99.5)
        },
        'Sumatera Selatan': {
            'lat_range': (-4.5, -2.5), 
            'lon_range': (103.0, 105.5)
        },

        # Kalimantan
        'Kalimantan Timur': {
            'lat_range': (-1.0, 2.0), 
            'lon_range': (116.0, 118.5)
        },

        # Sulawesi
        'Sulawesi Selatan': {
            'lat_range': (-6.2, -2.5), 
            'lon_range': (118.5, 120.5)
        },

        # Bali
        'Bali': {
            'lat_range': (-8.8, -8.2), 
            'lon_range': (114.5, 115.6)
        },

        # Papua
        'Papua': {
            'lat_range': (-4.5, -2.0), 
            'lon_range': (134.0, 141.0)
        }
    }

    
    # Income categories with typical ranges (in IDR)
    income_ranges = {
        'Low': (2000000, 8000000),      # 2-8 million IDR
        'Medium': (8000000, 20000000),   # 8-20 million IDR  
        'High': (20000000, 50000000)     # 20-50 million IDR
    }
    
    # Start date
    start_date = datetime(2006, 12, 16)
    
    # Generate data for 30 days with hourly intervals
    data = []
    
    # Create different customer profiles
    customer_profiles = {
        'nighttime': {'peak_hours': [22, 23, 0, 1, 2], 'base_consumption': 0.8, 'peak_multiplier': 2.5},
        'morning': {'peak_hours': [6, 7, 8, 9, 10], 'base_consumption': 1.0, 'peak_multiplier': 2.0},
        'daytime': {'peak_hours': [11, 12, 13, 14, 15], 'base_consumption': 1.2, 'peak_multiplier': 1.8},
        'evening': {'peak_hours': [17, 18, 19, 20, 21], 'base_consumption': 0.9, 'peak_multiplier': 2.2},
        'stable': {'peak_hours': list(range(24)), 'base_consumption': 1.1, 'peak_multiplier': 1.2}
    }
    
    # Generate 30 days of data
    for day in range(30):
        current_date = start_date + timedelta(days=day)
        
        for hour in range(24):
            current_datetime = current_date + timedelta(hours=hour)
            
            # Randomly select region
            region = random.choice(list(indonesian_regions.keys()))
            region_info = indonesian_regions[region]
            
            # Generate random coordinates within region bounds
            lat = np.random.uniform(region_info['lat_range'][0], region_info['lat_range'][1])
            lon = np.random.uniform(region_info['lon_range'][0], region_info['lon_range'][1])
            
            # Randomly select income category with some regional bias
            if region in ['Jakarta', 'Jawa Barat', 'Bali']:
                income_weights = [0.3, 0.4, 0.3]  # More high income in developed regions
            elif region in ['Papua', 'Kalimantan Timur']:
                income_weights = [0.5, 0.3, 0.2]  # More low income in remote regions
            else:
                income_weights = [0.4, 0.4, 0.2]  # Balanced distribution
            
            income_category = np.random.choice(['Low', 'Medium', 'High'], p=income_weights)
            
            # Generate income within category range
            income_range = income_ranges[income_category]
            income_idr = np.random.uniform(income_range[0], income_range[1])
            
            # Select customer profile based on region and income
            if income_category == 'High':
                profile_weights = [0.1, 0.2, 0.3, 0.3, 0.1]  # High income prefers daytime/evening
            elif income_category == 'Low':
                profile_weights = [0.3, 0.3, 0.1, 0.2, 0.1]  # Low income more nighttime/morning
            else:
                profile_weights = [0.2, 0.2, 0.2, 0.3, 0.1]  # Medium balanced
            
            profile_type = np.random.choice(list(customer_profiles.keys()), p=profile_weights)
            profile = customer_profiles[profile_type]
            
            # Calculate power consumption based on profile
            base_power = profile['base_consumption']
            
            # Add peak hour effect
            if hour in profile['peak_hours']:
                power_multiplier = profile['peak_multiplier']
            else:
                power_multiplier = 1.0
            
            # Add some randomness and seasonal effect
            seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * day / 30)  # Monthly cycle
            daily_factor = 1.0 + 0.05 * np.sin(2 * np.pi * hour / 24)   # Daily cycle
            random_factor = np.random.normal(1.0, 0.15)                   # Random noise
            
            # Income effect on consumption (higher income = higher base consumption)
            income_factor = {
                'Low': 0.8,
                'Medium': 1.0, 
                'High': 1.4
            }[income_category]
            
            # Final power calculation
            global_active_power = (base_power * power_multiplier * seasonal_factor * 
                                 daily_factor * random_factor * income_factor * 100)
            
            # Ensure positive values
            global_active_power = max(global_active_power, 10)
            
            # Create record
            record = {
                'Date': current_datetime.strftime('%Y-%m-%d'),
                'Time': current_datetime.strftime('%H:%M:%S'),
                'Global_active_power': round(global_active_power, 3),
                'Income_IDR': int(income_idr),
                'Income_Category': income_category,
                'Region': region,
                'Lat': round(lat, 6),
                'Lon': round(lon, 6)
            }
            
            data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = "hourly_household_power_consumption_with_demo.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Data sintetis berhasil dibuat!")
    print(f"📁 File: {output_file}")
    print(f"📊 Total records: {len(df)}")
    print(f"📅 Periode: {df['Date'].min()} sampai {df['Date'].max()}")
    print(f"🏢 Regions: {', '.join(df['Region'].unique())}")
    print(f"💰 Income categories: {', '.join(df['Income_Category'].unique())}")
    
    # Display sample data
    print(f"\n📋 Sample data:")
    print(df.head(10))
    
    # Display statistics
    print(f"\n📈 Statistik Konsumsi per Income Category:")
    income_stats = df.groupby('Income_Category')['Global_active_power'].agg(['mean', 'std', 'min', 'max'])
    print(income_stats.round(2))
    
    print(f"\n🗺️ Statistik per Region:")
    region_stats = df.groupby('Region').agg({
        'Global_active_power': 'mean',
        'Income_IDR': 'mean',
        'Income_Category': 'nunique'
    }).round(2)
    print(region_stats)
    
    return df

if __name__ == "__main__":
    # Generate the synthetic data
    synthetic_df = generate_synthetic_power_data()
    
    print("\n🎯 Data siap untuk digunakan dalam dashboard segmentasi!")
    print("Jalankan file segmentation dashboard untuk melihat hasil analisis.")