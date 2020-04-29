"""
Default setting up, dumping things in data/moses*, to be launched from root
"""

from data_processing import download_moses, get_selfies, add_properties, add_scores

if __name__=='__main__':
    download_moses.download_moses()
    get_selfies.add_selfies()
    add_properties.add_props()
    add_scores.add_scores()