from seleniumbase import Driver
from . import config


def create_stealth_driver(headless: bool = True) -> Driver:
    """
    Creates and returns a "supercharged" Selenium driver instance using seleniumbase's UC mode.
    """
    driver = Driver(
        uc=config.SELENIUM_CONFIG["uc_driver"],
        headless=headless,
        agent=None,
    )

    width, height = map(int, config.SELENIUM_CONFIG["window_size"].split(','))
    driver.set_window_size(width, height)

    return driver