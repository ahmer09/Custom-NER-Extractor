# import custom modules
import html_parsing as hp

# define listing sources
Tech_Topics=[
    'https://medium.com/topic/artificial-intelligence',
    'https://medium.com/topic/blockchain',
    'https://medium.com/topic/cryptocurrency',
    'https://medium.com/topic/data-science',
    'https://medium.com/topic/machine-learning',
    'https://medium.com/topic/neuroscience',
    'https://medium.com/topic/programming',
    'https://medium.com/topic/self-driving-cars',
    'https://medium.com/topic/software-engineering',
    'https://medium.com/topic/technology',
    'https://medium.com/topic/startups']

# call the class
at = hp.ArticleTable(Tech_Topics)
multiple_articles_table_df = at.main() # Compute a Pandas dataframe to write into multiple_articles_table
multiple_articles_table_df.to_csv('file1.csv', index=False)