from bs4 import BeautifulSoup
import requests
import csv
import time

URL = "https://transcripts.foreverdreaming.org/viewtopic.php?f=431&t="
FILE_PATH = "simpsons_scripts_3.txt"

def get_script(script_id):
    """
    Makes request to URL and retrieves script
    @param script_id: ID of script
    """
    try:
        req = requests.get(URL + str(script_id))
        soup = BeautifulSoup(req.text, "lxml")
        script = soup.find_all('p')
        time.sleep(1)
        return script
    except:
        pass


if __name__ == "__main__":
    scripts = []
    with open(FILE_PATH, 'a+') as file:
        for id in range(21865, 22260):
            script = get_script(id)
            file.write(str(script))
            if script is not None:
                print("Script %i written to %s" % (id, FILE_PATH))
    file.close()
