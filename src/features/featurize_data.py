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
    X['email_in_top_five_domains'] = X['email_domain'].map(lambda x: x in ('gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'live.com')).astype(int)
    
    # featurize description contents:
    top_desc_fraud_words = ['vip', 'pour', 'party', 'code', 'club']
    for word in top_desc_fraud_words:
        X[word] = X['description'].str.contains(word).values.astype(int)

    return X