import datetime

def download_fama_french_five_factors_daily(start, end):
    """
    Download the Fama-French Five-Factor model data from the Ken French data library
    
    Parameters
    ----------
    start : datetime.date
        The start date for the data
    end : datetime.date
        The end date for the data
        
    Returns
    -------
    pandas.DataFrame
        The Fama-French Five Factors
    """
    # Import the pandas_datareader package
    import pandas_datareader.data as web

    # Use DataReader to fetch the Fama-French Five-Factor model data
    ff_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3_Daily', 'famafrench', start, end)

    # The Fama-French dataset can contain different tables. For example, '0' is typically the annual data,
    # '1' might be the monthly data, etc. Here we access the daily data.
    ff_daily = ff_factors[0]

    return ff_daily