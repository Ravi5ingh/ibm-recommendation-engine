import utility.util as ut
import models.extensions as ex

def get_article_to_interactions_mapping(interactions):
    """
    Get the mapping that gives you the number of interactions for each article
    :param interactions: The interactions data
    :return: The dictionary mapping
    """

    return __get_interaction_frequency_of__('article_id', interactions)

def get_email_to_interactions_mapping(interactions):
    """
    Get a mapping that give you the number of interactions for each email
    :param interactions: The user article interaction data
    :return: The dictionary mapping
    """

    return __get_interaction_frequency_of__('email', interactions)

def __get_interaction_frequency_of__(obj_of_interaction, interactions):
    """
    Get the frequency with which the object of interaction has been present in an interaction
    :param obj_of_interaction: The object of interaction
    :param interactions: The interactions data
    :return: The dictionary mapping of interaction frequency
    """

    print(f'Getting {obj_of_interaction} to interactions mapping...')
    email_interactions = ex.dictionary()
    total = ut.row_count(interactions)
    for index, row in interactions.iterrows():
        if row[obj_of_interaction] in email_interactions:
            email_interactions[row[obj_of_interaction]] += 1
        else:
            email_interactions[row[obj_of_interaction]] = 1
        ut.update_progress(index, total)
    print('\n')

    return email_interactions