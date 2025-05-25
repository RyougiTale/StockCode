import os
import pandas as pd
import baostock as bs
from datetime import datetime, timedelta

class IndexKLineLoader:
    def __init__(self, data_dir='./data/index_kline_data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.bs_logged_in = False

    def _get_cache_filepath(self, index_code, frequency):
        return os.path.join(self.data_dir, f"{index_code.replace('.', '_')}_{frequency}.csv")

    def login_bs(self):
        if not self.bs_logged_in:
            lg = bs.login()
            if lg.error_code == '0':
                self.bs_logged_in = True
                print("Baostock login successful for IndexKLineLoader.")
            else:
                print(f"Baostock login failed for IndexKLineLoader: {lg.error_msg}")
                self.bs_logged_in = False
        return self.bs_logged_in

    def logout_bs(self):
        if self.bs_logged_in:
            bs.logout()
            self.bs_logged_in = False
            print("Baostock logout successful for IndexKLineLoader.")

    def _fetch_index_k_data_from_bs(self, index_code, start_date_str, end_date_str, frequency, fields):
        """
        Fetches index K-line data from Baostock.
        end_date_str is exclusive for Baostock API.
        """
        if not self.login_bs():
            return None

        print(f"Fetching index K-line data for {index_code} ({frequency}) from Baostock: {start_date_str} to {end_date_str} (exclusive end)...")
        rs = bs.query_history_k_data_plus(
            code=index_code,
            fields=fields,
            start_date=start_date_str,
            end_date=end_date_str, # API end_date is exclusive
            frequency=frequency,
            adjustflag="3" # Default to no adjustment for indices usually, 3=不复权
        )
        
        if rs.error_code != '0':
            print(f"Baostock query_history_k_data_plus for {index_code} failed: {rs.error_msg}")
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            print(f"No K-line data returned from Baostock for {index_code} ({frequency}) for the period.")
            return pd.DataFrame()

        result_df = pd.DataFrame(data_list, columns=rs.fields)
        return result_df

    def _process_index_k_data_df(self, df):
        if df.empty:
            return df
            
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg']
        for col in numeric_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        df_copy.sort_values(by='date', inplace=True)
        df_copy.reset_index(drop=True, inplace=True)
        return df_copy

    def load_index_k_data(self, 
                          index_code="sh.000300", 
                          start_date="2005-01-01", 
                          end_date=None, 
                          frequency="d", 
                          fields="date,code,open,high,low,close,preclose,volume,amount,pctChg", 
                          use_cache=True):
        """
        Loads index K-line data.
        start_date and end_date for this method are inclusive.
        """
        target_start_dt = pd.to_datetime(start_date)
        if end_date is None:
            target_end_dt = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        else:
            target_end_dt = pd.to_datetime(end_date)

        cache_filepath = self._get_cache_filepath(index_code, frequency)
        cached_df = pd.DataFrame()

        if use_cache and os.path.exists(cache_filepath):
            try:
                cached_df = pd.read_csv(cache_filepath)
                if 'date' in cached_df.columns:
                    cached_df['date'] = pd.to_datetime(cached_df['date'])
                else:
                    print(f"Cache file {cache_filepath} is missing 'date'. Re-fetching all.")
                    cached_df = pd.DataFrame()
                if cached_df.empty:
                    print(f"Cache file {cache_filepath} is empty.")
                else:
                    print(f"Loaded {len(cached_df)} records from K-line cache: {cache_filepath}")
            except Exception as e:
                print(f"Error reading K-line cache file {cache_filepath}: {e}. Will fetch all data.")
                cached_df = pd.DataFrame()
        
        fetch_start_dt = target_start_dt
        
        if not cached_df.empty:
            cached_df.sort_values(by='date', inplace=True)
            last_cached_date = cached_df['date'].max()
            
            # Next day to fetch starts after the last cached date
            # For daily data, this is simple. For weekly/monthly, Baostock provides data on last trading day of period.
            # Assuming daily for simplicity in this increment logic.
            next_day_to_fetch = last_cached_date + timedelta(days=1) 
            
            if next_day_to_fetch > target_end_dt:
                print(f"K-line cache for {index_code} ({frequency}) is up to date. No new data to fetch.")
                final_df = cached_df[(cached_df['date'] >= target_start_dt) & 
                                     (cached_df['date'] <= target_end_dt)].copy()
                final_df.reset_index(drop=True, inplace=True)
                return final_df
            
            fetch_start_dt = next_day_to_fetch
            print(f"K-line cache for {index_code} ({frequency}) up to {last_cached_date.strftime('%Y-%m-%d')}. Fetching new data from {fetch_start_dt.strftime('%Y-%m-%d')} to {target_end_dt.strftime('%Y-%m-%d')}.")
        else:
            print(f"No K-line cache for {index_code} ({frequency}) found or cache is empty. Fetching data from {fetch_start_dt.strftime('%Y-%m-%d')} to {target_end_dt.strftime('%Y-%m-%d')}.")

        # API's end_date is exclusive, so add one day to target_end_dt for the call
        api_end_date_str = (target_end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        
        if fetch_start_dt > target_end_dt : # If fetch_start_dt is already past the inclusive target_end_dt
             print(f"Calculated fetch start date {fetch_start_dt.strftime('%Y-%m-%d')} is after target end date {target_end_dt.strftime('%Y-%m-%d')}. No data to fetch.")
             new_data_df_processed = pd.DataFrame()
        else:
            new_data_df_raw = self._fetch_index_k_data_from_bs(
                index_code, 
                fetch_start_dt.strftime("%Y-%m-%d"), 
                api_end_date_str, # Exclusive end for API
                frequency, 
                fields
            )
            if new_data_df_raw is not None and not new_data_df_raw.empty:
                new_data_df_processed = self._process_index_k_data_df(new_data_df_raw)
            else:
                new_data_df_processed = pd.DataFrame()

        if not new_data_df_processed.empty:
            combined_df = pd.concat([cached_df, new_data_df_processed], ignore_index=True)
            # Ensure 'date' column is present before drop_duplicates
            if 'date' in combined_df.columns:
                combined_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
                combined_df.sort_values(by='date', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
        else:
            combined_df = cached_df.copy()

        if not combined_df.empty:
            try:
                combined_df.to_csv(cache_filepath, index=False, encoding='utf-8')
                print(f"Updated K-line cache file: {cache_filepath} with {len(combined_df)} records.")
            except Exception as e:
                print(f"Error writing to K-line cache file {cache_filepath}: {e}")
        
        final_df = combined_df[(combined_df['date'] >= target_start_dt) & 
                               (combined_df['date'] <= target_end_dt)].copy()
        final_df.reset_index(drop=True, inplace=True)
        
        self.logout_bs()
        return final_df

if __name__ == '__main__':
    loader = IndexKLineLoader()
    
    # Example: Load default CSI 300 index data
    # csi300_df = loader.load_index_k_data(index_code="sh.000300", start_date="2007-01-01")
    csi300_df = loader.load_index_k_data(index_code="sh.510300", start_date="2007-01-01")
    if csi300_df is not None and not csi300_df.empty:
        print("\nLoaded CSI 300 (sh.000300) K-line Data:")
        print(csi300_df.head())
        print("...")
        print(csi300_df.tail())
        print(f"Total records: {len(csi300_df)}")
    else:
        print("No CSI 300 K-line data loaded.")

    # Example: Load Shanghai Composite Index (sh.000001) data for a specific period
    # sh_comp_df = loader.load_index_k_data(index_code="sh.000001", start_date="2023-01-01", end_date="2023-03-31")
    # if sh_comp_df is not None and not sh_comp_df.empty:
    #     print("\nLoaded Shanghai Composite (sh.000001) K-line Data (2023 Q1):")
    #     print(sh_comp_df)
    # else:
    #     print("No Shanghai Composite K-line data loaded for the period.")
    
    
    
    
    
    
    
# python Util/IndexKLineLoader.py --code sh.000300 --freq d 