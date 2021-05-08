from city import City


def read_input(path: str) -> list[City]:
    cities = []
    with open(path, "r") as input_file:
        for line in input_file:
            x, y = line.split(" ")
            city = City(x=float(x), y=float(y))
            cities.append(city)
    return cities
