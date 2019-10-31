import numpy as np

def featurize_data(X):
    X['event_duration'] = X['event_end'] - X['event_start']
    X['time_to_create'] = X['event_published'] - X['event_created']
    X['org_desc_exists'] = np.where(X['org_desc'] != '', 1, 0)
    X['org_name_exists'] = np.where(X['org_name'] != '', 1, 0)
    X['num_previous_payouts'] = X['previous_payouts'].map(lambda x: len(x))
    X['num_ticket_types'] = X['ticket_types'].map(lambda x: len(x))
    X['venue_address_exists'] = np.where(X['venue_address'] != '', 1, 0)
    X['venue_name_exists'] = np.where(X['venue_name'] != '', 1, 0)

    return X