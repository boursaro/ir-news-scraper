from collections import defaultdict
import json
from time import sleep

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By


def select_elements(driver, selector):
    ref = selector["ref"].replace('\\', '')
    if selector["type"] == "single":
        return [driver.find_element(By.XPATH, ref)]
    elif selector["type"] == "group":
        return driver.find_elements(By.XPATH, ref + selector["target"])


def create_result_storages(storages):
    return {
        storage["id"]: []
        for storage in storages
    }


def apply_actions(driver, actions, result_storages, current_storage=None):
    last_elements = None

    for action in actions:
        type_ = action["type"]
        selector = action["selector"]
        linked_actions = action.get("linked_actions", [])
        result_storage_id = action.get("result_storage_id", None)

        elements = select_elements(driver, selector) or last_elements
        last_elements = elements

        match type_:
            case "click":
                for element in elements:
                    element.click()

                    if result_storage_id:
                        current_storage = defaultdict(list)
                        result_storages.append(current_storage)

                    apply_actions(driver, linked_actions, result_storages, current_storage)

            case "goto":
                for element in elements:
                    target_url = element.get_attribute("href")
                    driver.get(target_url)

                    if result_storage_id:
                        current_storage = defaultdict(list)
                        result_storages.append(current_storage)

                    apply_actions(driver, linked_actions, result_storages, current_storage)

                    driver.back()

            case "extract":
                result_storage_key = action["result_storage_key"]
                for element in elements:
                    current_storage[result_storage_key].append(element.text)


def run_instructions(instructions):
    url = instructions["url"]
    actions = instructions["actions"]
    storages = instructions["storages"]

    chrome_options = webdriver.ChromeOptions()
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.get(url)

    # result_storages = create_result_storages(storages)
    result_storages = []
    apply_actions(driver, actions, result_storages, current_storage=None)

    driver.close()

    return result_storages


def main():
    with open("instruction_sample.json") as f:
        data = json.loads(f.read())

    result_storage = run_instructions(data)

    with open("scrapper_result.json", "wt", encoding="utf-8") as f:
        f.write(json.dumps(result_storage, ensure_ascii=False))


if __name__ == "__main__":
    main()