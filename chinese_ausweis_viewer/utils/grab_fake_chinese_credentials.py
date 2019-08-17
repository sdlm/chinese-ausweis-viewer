from typing import List
from datetime import datetime

import requests
from lxml import html

FAKE_CREDENTIALS_URL = 'http://www.myfakeinfo.com/nationalidno/get-china-citizenidandname.php'
TABLE_XPATH = '/html/body/div[1]/div[3]/div[1]/table/tbody/tr'

FIELDS_MAPPING = {
    0: 'name',
    1: 'id',
    2: 'sex',
    3: 'birthday',
    5: 'address',
}
GENDER_MAPPING = {
    'male': '男',
    'female': '女',
}


def get_chinese_creds(count: int = 20) -> List[dict]:
    """
    format:
        name: str
        sex: str
        nationality: str
        birthday: date
        address: str
        id: str
    """

    persons = []
    while True:
        response = requests.get(FAKE_CREDENTIALS_URL)
        tree = html.fromstring(response.text)
        for row in tree.xpath(TABLE_XPATH):
            elems = row.findall('td')
            temp = {
                v: elems[k].text
                for k, v in FIELDS_MAPPING.items()
            }
            persons.append(
                dict(
                    name=temp['name'],
                    sex=GENDER_MAPPING[temp['sex']],
                    nationality='汉',
                    birthday=datetime.strptime(temp['birthday'], "%Y%m%d").date(),
                    address=temp['address'],
                    id=temp['id'],
                )
            )
            if len(persons) == count:
                break
        if len(persons) == count:
            break

    return persons
