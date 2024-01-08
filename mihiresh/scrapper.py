import requests
from bs4 import BeautifulSoup

def scrape_wikipedia(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all the section headers (assumes they are in p tags)
        section_headers = soup.find_all('h2')
        section_paras = soup.find_all('p')
        section_lists = soup.find_all('li')

        # Print the titles of the sections
        # for header in section_headers:
        #    print(header.text.strip())
        text= """"""
        for header in section_paras:
            #print(header.text.strip())
            text += header.text.strip()
        #for header in section_lists:
        #    print(header.text.strip())

# Split the text into lines
        lines = text.split('\n')

# Select the first 15 lines
        first_15_lines = lines[:15]

# Join the selected lines into a single string
        result = '\n'.join(first_15_lines)

# Print the result
        print(result)

    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")



keyword='Ganpati'
# Example usage: scraping the Python programming language Wikipedia page
wikipedia_url = 'https://en.wikipedia.org/wiki/'+keyword
scrape_wikipedia(wikipedia_url)
