import utility.util as ut

def get_article_id_frequency(articles):
    """
    Get the frequency with which each article Id appears
    :param articles: The articles data
    :return: The frequency mapping
    """

    print('Scanning articles for dupes...')
    total = ut.row_count(articles)
    article_frequency_mapping = {}
    for index, row in articles.iterrows():
        if row['article_id'] in article_frequency_mapping:
            article_frequency_mapping[row['article_id']] += 1
        else:
            article_frequency_mapping[row['article_id']] = 1
        ut.update_progress(index, total)
    print('\n')

    for article_id, frequency in ut.sorted_dictionary(article_frequency_mapping, ascending=False):
        print(f'Article Id: {article_id} appeared {frequency} times')
        input()

