import random
import json


def create_synth_dataset(with_misc_classes=True):

    animals = [
        "dog", "cat", "horse", "spider", "butterfly", "chicken", "sheep", "cow", "squirrel", "elephant",
        "lion", "tiger", "bear", "snake", "frog", "fish", "eagle", "rabbit", "monkey", "fox",
        "wolf", "deer", "owl", "penguin", "kangaroo", "panda", "zebra", "leopard", "goat", "duck",
        "parrot", "bee", "camel", "dolphin", "whale", "mouse", "hawk", "peacock", "seal", "raccoon",
        "mountain lion", "sea turtle", "polar bear", "snow leopard", "golden eagle",
        "cheetah", "hippopotamus", "hummingbird", "rhinoceros", "grizzly bear", "killer whale",
        "rattlesnake", "koala", "octopus", "jellyfish", "alligator", "chimpanzee", "vulture",
        "armadillo", "manatee", "badger", "hyena", "iguana", "gorilla", "sloth", "puffin",
        "chameleon", "wombat", "fennec fox", "red panda", "tuna", "stork", "shrimp", "coyote",
        "scorpion", "gull", "flamingos", "caterpillar", "lobster", "crab", "wasp", "snail", "slug"
    ]

    animal_plurals = {
        "fish": "fish", "sheep": "sheep", "deer": "deer", "mouse": "mice",
        "octopus": "octopuses", "hippopotamus": "hippopotamuses",
        "flamingos": "flamingos"
    }

    def get_plural(noun):
        if noun in animal_plurals:
            return animal_plurals[noun]
        if noun.endswith('y'):
            return noun[:-1] + 'ies'
        if noun.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return noun + 'es'
        return noun + 's'

    people = [
        "John", "Maria", "Alice", "David", "Michael", "Emma", "Olivia", "Daniel", "Sophia", "James",
        "Ethan", "Grace", "Lucas", "Chloe", "Benjamin", "Hannah", "Jacob", "Liam", "Ava", "Noah",
        "Anna Smith", "Robert Brown", "Emily Johnson", "Sarah Connor", "Tom Cruise", "Elena Rodriguez",
        "Dr. Peterson", "Professor Lee", "King Charles", "Queen Elizabeth",
        "Mr. Henderson", "Mrs. Williams", "Sergeant Miller", "Officer Jones", "President Bush"
    ]

    places = [
        "Paris", "London", "New York", "Tokyo", "Berlin", "Madrid", "Sydney", "Rome", "Toronto", "Cairo",
        "Lisbon", "Moscow", "Seoul", "Dubai", "Amsterdam", "Prague", "Dublin", "Vienna", "Budapest", "Oslo",
        "Los Angeles", "San Francisco", "Cape Town", "Geneva", "Stockholm", "Rio de Janeiro", "Helsinki",
        "Athens", "Mumbai", "Shanghai", "Mexico City", "Mount Everest", "Sahara Desert", "Amazon Rainforest"
    ]

    orgs = [
        "Google", "Microsoft", "NASA", "UNICEF", "Apple", "Amazon", "Tesla", "Meta", "Harvard", "Netflix",
        "IBM", "Intel", "Oxford", "Samsung", "Airbnb", "SpaceX", "MIT", "Spotify", "Adobe", "Huawei",
        "World Health Organization", "National Geographic", "Red Cross", "United Nations", "Greenpeace",
        "Toyota", "Siemens", "The British Museum", "Acme Corp", "General Motors", "The World Bank"
    ]

    objects = [
        "bridge", "car", "house", "river", "forest", "school", "phone", "mountain", "book", "computer",
        "statue", "museum", "garden", "tower", "road", "library", "market", "castle", "tree", "temple",
        "apartment", "skyscraper", "train station", "motorcycle", "ship", "pyramid", "ruins", "tractor"
    ]

    adjectives = [
        "big", "small", "fierce", "playful", "cunning", "swift", "white", "black", "huge", "tiny",
        "majestic", "agile", "rare", "wounded", "young", "old", "tropical", "arctic", "dangerous",
        "colorful", "quiet", "noisy", "striped", "spotted", "venomous",
        "scaly", "furry", "feathered", "aquatic", "nocturnal", "exotic", "enormous", "ancient", "fast"
    ]


    templates_animals = [
        "There is a {adj} {animal} in the picture.",
        "A {animal} was seen near the {object} yesterday.",
        "They found a {adj} {animal} in the {place} forest.",
        "The {adj} {animal} is running across the field.",
        "A group of {animal_plural} was spotted resting near the lake.",
        "During the expedition, researchers observed a rare {animal} in the mountains of {place} and recorded its calls.",
        "{person} said they once owned a {animal} when they lived in {place}, but now they have a {animal2}.",
        "{org} started a campaign to protect the endangered {adj} {animal} population, which faces threats from pollution.",
        "The documentary crew filmed the {animal} migration near the {object} in {place} just before winter.",
        "A {animal} attacked a herd of {animal2_plural} close to the {object}.",
        "The {animal} is native to the {adj} forests near {place}, living alongside the {animal2}.",
        "A {young_adj} {animal} was rescued by {org} volunteers after falling into the {object}.",
        "The {animal} and the {animal2} share the same habitat, often competing for resources.",
        "Some people keep a {animal} as a pet, but others prefer a {animal2} or even a {animal3}.",
        "The {adj} {animal} was seen hunting near the {object} in {place} at dawn.",
        "A painting of a {animal} was found in an ancient cave in {place} which showed hunting rituals.",
        "Look at that {adj} {animal}!",
        "The scientists studied the behavior of the {adj} {animal_plural} in the wild.",
        "The {animal_multiword} is a fascinating creature, known for its unique hunting style.",
        "The diet of the {animal_multiword} mainly consists of fish, but they also eat {animal2}.",
        "The {animal} is recognized as an apex predator in its biome near the {object}.",
        "{animal_plural} are generally considered harmless unless provoked.",
        "The {animal} that attacked the sheep belonged to a pack near {place}.",
        "If you look closely, you can spot a {adj} {animal} hidden amongst the trees.",
        "Neither the {animal} nor the {animal2} could survive the cold winter in {place}.",
        "The survival rate of the {young_adj} {animal} depends on the protection offered by {org}."
    ]

    templates_misc = [
        "{person} works for {org} in {place}.",
        "{person} visited {place} last summer, while {person2} was working at {org}.",
        "{org} opened a new office in {place} with support from {org2} to expand their operations.",
        "{person} and {person2} met at the {object} museum and discussed the new exhibit.",
        "{org} announced a partnership with {org2} to fund research into clean energy.",
        "{place} is famous for its old {adj} {object}, which dates back to the 16th century.",
        "{org} supports education in {place} and other nearby regions, donating books to {object}.",
        "Many tourists come to {place} every year to see the {object}, including {person}.",
        "{person} graduated from {org} and later moved to {place} to pursue a career in technology.",
        "{org} launched an initiative to rebuild the {object}s destroyed in {place} following the storm.",
        "{org} held a press conference in {place} together with {org2}'s CEO, {person}.",
        "{person} lives near the {object} in the city of {place}.",
        "{person} wrote a book about their time at {org} and life in {place}.",
        "{org} collaborated with {org2} to fund a new {adj} {object} in {place}.",
        "The annual conference hosted by {org} took place in the famous {place}, attended by {person} and {person2}.",
        "{org}’s headquarters in {place} employs thousands of people, including {person}.",
        "{place}’s mayor thanked {org} for supporting local infrastructure and rebuilding the {object}.",
        "Key figures like {person}, {person2} spoke at the event hosted by {org} in {place}.",
        "It was {person} who first suggested that {org} move its operations to {place}.",
        "Before working at {org}, {person2} was a student in {place}.",
        "We need to contact {person} at {org} regarding the shipment to {place}."
    ]

    templates_neutral = [
        "The weather is very nice today.",
        "He bought a new pair of shoes yesterday evening.",
        "The book on the table is mine, not yours.",
        "They are watching a movie together at home.",
        "I made some coffee this morning before starting work.",
        "She cleaned the kitchen and went outside for a walk.",
        "This software update fixed many critical issues.",
        "We should go for a walk in the evening tomorrow.",
        "The road is closed because of unexpected maintenance.",
        "There are many stars visible tonight in the clear sky.",
        "She quickly finished her assignment before the deadline.",
        "Please turn off the lights when you leave the room.",
        "The old wooden door creaked loudly as the wind blew through the empty corridor.",
        "If the rain stops by noon, we will definitely go to the park later.",
        "She meticulously organized all the documents into separate, clearly labeled folders."
    ]


    num_animals = 5000
    num_misc = 4500
    num_neutral = 500

    dataset = []


    def tag_entity(tokens, entity_phrase, label_prefix, labels):
        phrase_tokens = entity_phrase.split()

        clean_tokens = [t.strip(".,!?") for t in tokens]

        for i in range(len(clean_tokens) - len(phrase_tokens) + 1):
            if clean_tokens[i:i + len(phrase_tokens)] == phrase_tokens:
                if labels[i] == "O":
                    labels[i] = f"B-{label_prefix}"
                    for j in range(1, len(phrase_tokens)):
                        if labels[i + j] == "O":
                            labels[i + j] = f"I-{label_prefix}"
                return True
        return False




    for _ in range(num_animals):
        template = random.choice(templates_animals)


        animal = random.choice(animals)
        animal2 = random.choice(animals)
        person = random.choice(people)
        place = random.choice(places)
        org = random.choice(orgs)
        obj = random.choice(objects)
        adj = random.choice(adjectives)


        mapping = {
            'animal': animal,
            'animal2': animal2,
            'animal_plural': get_plural(animal),
            'animal2_plural': get_plural(animal2),
            'animal_multiword': random.choice([a for a in animals if ' ' in a] + [animal]),
            'person': person,
            'person2': random.choice(people),
            'place': place,
            'org': org,
            'org2': random.choice(orgs),
            'object': obj,
            'adj': adj,
            'young_adj': random.choice(["young", "old"])
        }

        try:
            sentence = template.format(**mapping)
        except KeyError as e:
            continue

        tokens = sentence.replace(".", " .").replace(",", " ,").replace("!", " !").replace("?", " ?").split()
        labels = ["O"] * len(tokens)


        entities_to_tag = [
            (mapping.get('animal_multiword'), "ANIMAL"),
            (mapping.get('animal'), "ANIMAL"),
            (mapping.get('animal2'), "ANIMAL"),
        ]

        if with_misc_classes:
            entities_to_tag.extend([(mapping.get('person'), "MISC"),
            (mapping.get('person2'), "MISC"),
            (mapping.get('place'), "MISC"),
            (mapping.get('org'), "MISC"),
            (mapping.get('org2'), "MISC")])


        if '{animal_plural}' in template:
            entities_to_tag.append((mapping['animal_plural'], "ANIMAL"))
        if '{animal2_plural}' in template:
            entities_to_tag.append((mapping['animal2_plural'], "ANIMAL"))

        for ent, tag in entities_to_tag:
            if ent:
                tag_entity(tokens, ent, tag, labels)

        dataset.append({"tokens": tokens, "labels": labels})

    for _ in range(num_misc):
        template = random.choice(templates_misc)

        mapping = {
            'person': random.choice(people),
            'person2': random.choice(people),
            'org': random.choice(orgs),
            'org2': random.choice(orgs),
            'place': random.choice(places),
            'object': random.choice(objects),
            'adj': random.choice(adjectives)
        }

        sentence = template.format(**mapping)
        tokens = sentence.replace(".", " .").replace(",", " ,").replace("!", " !").replace("?", " ?").split()
        labels = ["O"] * len(tokens)


        if with_misc_classes:
            entities_to_tag = [
                (mapping['person'], "MISC"),
                (mapping['person2'], "MISC"),
                (mapping['org'], "MISC"),
                (mapping['org2'], "MISC"),
                (mapping['place'], "MISC"),
            ]

            for ent, tag in entities_to_tag:
                if ent:
                    tag_entity(tokens, ent, tag, labels)

        dataset.append({"tokens": tokens, "labels": labels})

    for _ in range(num_neutral):
        sentence = random.choice(templates_neutral)
        tokens = sentence.replace(".", " .").replace(",", " ,").replace("!", " !").replace("?", " ?").split()
        labels = ["O"] * len(tokens)
        dataset.append({"tokens": tokens, "labels": labels})

    random.shuffle(dataset)
    if with_misc_classes:
        path = "animal_ner_synthetic_5_classes.jsonl"
    else:
        path = "animal_ner_synthetic_3_classes.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} greatly expanded synthetic samples to {path}")


if __name__ == '__main__':
    create_synth_dataset()