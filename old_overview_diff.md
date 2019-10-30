    1. Only unpickle the model once
    2. Only connect to the database once.
    
    Do both in a `if __name__ == '__main__':` block before you call `app.run()` and you can refer to these top-level global variables from within the function. This may require some re-architecting of your prediction module.

    The individual example will no longer be coming from a local file, but instead you will get it by making a request to a server that will give you a data point as a string, which you can parse into JSON. 
You can use `json.loads()` to parse a string to json, which is the reverse process of `json.dumps()`. You'll still need to vectorize it, predict, and store the example and prediction in the database.

### Step 6: Get "live" data

We've set up a service for you that will send out "live" data so that you can see that your app is really working.

To use this service, you will need to make a request to our secure server. It gives a maximum of the 10 most recent datapoints, ordered by `sequence_number`. New datapoints come in every few minutes.

*Warning: you will need to implement the save_to_database method.*

```python
class EventAPIClient:
    """Realtime Events API Client"""
    
    def __init__(self, first_sequence_number=0,
                 api_url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/',
                 api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC',
                 db = None):
        """Initialize the API client."""
        self.next_sequence_number = first_sequence_number
        self.api_url = api_url
        self.api_key = api_key
        
    def save_to_database(self, row):
        """Save a data row to the database."""
        print("Received data:\n" + repr(row) + "\n")  # replace this with your code

    def get_data(self):
        """Fetch data from the API."""
        payload = {'api_key': self.api_key,
                   'sequence_number': self.next_sequence_number}
        response = requests.post(self.api_url, json=payload)
        data = response.json()
        self.next_sequence_number = data['_next_sequence_number']
        return data['data']
    
    def collect(self, interval=30):
        """Check for new data from the API periodically."""
        while True:
            print("Requesting data...")
            data = self.get_data()
            if data:
                print("Saving...")
                for row in data:
                    self.save_to_database(row)
            else:
                print("No new data received.")
            print(f"Waiting {interval} seconds...")
            time.sleep(interval)


# Usage Example

client = EventAPIClient()
client.collect()
```

1. Write a function that periodically fetches new data, generates a predicted fraud probability, and saves it to your database (after verifying that the data hasn't been seen before).