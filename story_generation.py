import random
import json

random.seed(67)

OBJECTS = ["book", "pen", "notebook", "mug", "key"]
LOCATIONS = ["table", "shelf", "drawer", "backpack", "desk", "counter"]
NAMES = [
    "Diya",
    "Ali",
    "Nithin",
    "Kaviya",
    "Shreya",
    "Shrujal",
    "Arshika",
    "Yamha",
    "Manasvini",
    "Shrimayi",
    "Lipika",
    "Soha",
    "Puneet",
]

DISTRACTOR_SENTENCES = [
    "{name} likes reading.",
    "{name} is a chill guy.",
    "{name} enjoys coffee in the morning.",
    "{name} has been busy all day.",
    "{name} forgot to eat lunch.",
    "{name} is wearing a black shirt.",
    "{name} smiled and walked away.",
    "{name} was thinking about dinner.",
]


def move_sentence(name, obj, location):
    templates = [
        "{name} moves the {obj} to the {location}.",
        "{name} places the {obj} on the {location}.",
        "{name} puts the {obj} on the {location}.",
    ]
    return random.choice(templates).format(name=name, obj=obj, location=location)


def distractor_sentence(name):
    return random.choice(DISTRACTOR_SENTENCES).format(name=random.choice(NAMES))


def generate_distractor(num_transfers, num_distractors=2):
    obj = random.choice(OBJECTS)
    names = random.sample(NAMES, min(num_transfers + 1, len(NAMES)))
    locations = random.sample(LOCATIONS, min(num_transfers + 1, len(LOCATIONS)))

    sentences = []

    # initial placement
    sentences.append(move_sentence(names[0], obj, locations[0]))
    current_location = locations[0]
    old_locations = [locations[0]]

    # transfers
    for i in range(1, num_transfers + 1):
        name = names[i] if i < len(names) else random.choice(NAMES)
        loc = (
            locations[i]
            if i < len(locations)
            else random.choice([l for l in LOCATIONS if l != current_location])
        )
        sentences.append(move_sentence(name, obj, loc))
        old_locations.append(current_location)
        current_location = loc

    # insert distractors that mention an old location
    for _ in range(num_distractors):
        if old_locations:
            old_loc = (
                random.choice(old_locations[:-1])
                if len(old_locations) > 1
                else old_locations[0]
            )
            trap_name = random.choice(NAMES)
            sentences.append(f"{trap_name}'s favorite spot is the {old_loc}.")

    random.shuffle(
        sentences[num_transfers + 1 :]
    )  # shuffle only the distractors, keep moves in order

    sentences.append(f"Where is the {obj}?")

    return {
        "type": "distractor",
        "num_transfers": num_transfers,
        "num_location_distractors": num_distractors,
        "num_random_distractors": 0,
        "story": " ".join(sentences),
        "object": obj,
        "answer": current_location,
    }


def generate_red_herring(num_transfers, num_distractors=1):
    obj = random.choice(OBJECTS)
    other_obj = random.choice([o for o in OBJECTS if o != obj])

    names = random.sample(NAMES, min(num_transfers + 2, len(NAMES)))
    locations = random.sample(LOCATIONS, min(num_transfers + 2, len(LOCATIONS)))

    sentences = []

    # initial placement
    sentences.append(move_sentence(names[0], obj, locations[0]))
    current_location = locations[0]
    old_locations = [locations[0]]

    # transfers of target object
    for i in range(1, num_transfers + 1):
        name = names[i] if i < len(names) else random.choice(NAMES)
        loc = (
            locations[i]
            if i < len(locations)
            else random.choice([l for l in LOCATIONS if l != current_location])
        )
        sentences.append(move_sentence(name, obj, loc))
        old_locations.append(current_location)
        current_location = loc

    # red herring: move a different object to an old location
    red_herring_name = (
        names[-1] if len(names) > num_transfers + 1 else random.choice(NAMES)
    )
    red_herring_loc = (
        random.choice(old_locations[:-1])
        if len(old_locations) > 1
        else old_locations[0]
    )
    sentences.append(move_sentence(red_herring_name, other_obj, red_herring_loc))

    # optional extra distractors
    for _ in range(num_distractors):
        sentences.append(distractor_sentence(random.choice(NAMES)))

    sentences.append(f"Where is the {obj}?")

    return {
        "type": "red_herring",
        "num_transfers": num_transfers,
        "num_location_distractors": 1,  # always 1 red herring object move
        "num_random_distractors": num_distractors,
        "story": " ".join(sentences),
        "object": obj,
        "answer": current_location,
    }


# option 3: manually written reversal stories
REVERSAL_STORIES = [
    {
        "type": "reversal",
        "num_transfers": 2,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Diya puts the book on the table. Ali moves the book to the shelf. Nithin moves the book back. Where is the book?",
        "object": "book",
        "answer": "table",
    },
    {
        "type": "reversal",
        "num_transfers": 2,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Kaviya places the mug on the counter. Shreya moves the mug to the drawer. Shrujal moves the mug back. Where is the mug?",
        "object": "mug",
        "answer": "counter",
    },
    {
        "type": "reversal",
        "num_transfers": 3,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Arshika puts the pen on the desk. Yamha moves the pen to the backpack. Manasvini moves the pen to the shelf. Shrimayi moves the pen back. Where is the pen?",
        "object": "pen",
        "answer": "backpack",
    },
    {
        "type": "reversal",
        "num_transfers": 2,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Nithin puts the key on the counter. Soha moves the key to the backpack. Lipika moves the key back. Where is the key?",
        "object": "key",
        "answer": "counter",
    },
    {
        "type": "reversal",
        "num_transfers": 2,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Shrujal places the notebook on the desk. Yamha moves the notebook to the drawer. Ali moves the notebook back. Where is the notebook?",
        "object": "notebook",
        "answer": "desk",
    },
    {
        "type": "reversal",
        "num_transfers": 2,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Arshika puts the mug on the shelf. Manasvini moves the mug to the counter. Diya moves the mug back. Where is the mug?",
        "object": "mug",
        "answer": "shelf",
    },
    {
        "type": "reversal",
        "num_transfers": 3,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Soha places the book on the counter. Nithin moves the book to the desk. Kaviya moves the book to the shelf. Shreya moves the book back. Where is the book?",
        "object": "book",
        "answer": "desk",
    },
    {
        "type": "reversal",
        "num_transfers": 3,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Ali puts the mug on the backpack. Shrujal moves the mug to the counter. Shrimayi moves the mug to the table. Lipika moves the mug back. Where is the mug?",
        "object": "mug",
        "answer": "counter",
    },
    {
        "type": "reversal",
        "num_transfers": 3,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Yamha places the pen on the shelf. Arshika moves the pen to the backpack. Soha moves the pen to the drawer. Nithin moves the pen back. Where is the pen?",
        "object": "pen",
        "answer": "backpack",
    },
    {
        "type": "reversal",
        "num_transfers": 2,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Shrimayi puts the key on the table. Kaviya moves the key to the shelf. Manasvini moves the key back. Where is the key?",
        "object": "key",
        "answer": "table",
    },
]

CONTROL_STORIES = [
    {
        "type": "control",
        "num_transfers": 0,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Diya puts the book on the table. Where is the book?",
        "object": "book",
        "answer": "table",
    },
    {
        "type": "control",
        "num_transfers": 0,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Ali places the mug on the shelf. Where is the mug?",
        "object": "mug",
        "answer": "shelf",
    },
    {
        "type": "control",
        "num_transfers": 0,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Nithin puts the keys on the counter. Where is the keys?",
        "object": "keys",
        "answer": "counter",
    },
    {
        "type": "control",
        "num_transfers": 0,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Kaviya moves the pen to the drawer. Where is the pen?",
        "object": "pen",
        "answer": "drawer",
    },
    {
        "type": "control",
        "num_transfers": 0,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": "Shreya places the notebook on the desk. Where is the notebook?",
        "object": "notebook",
        "answer": "desk",
    },
]


def generate_dataset():
    dataset = []

    # distractor stories: ~25 stories, spread across 1-4 transfers
    for num_transfers in [0, 1, 2, 3, 4]:
        for _ in range(6):
            dataset.append(
                generate_distractor(num_transfers, num_distractors=random.randint(1, 3))
            )
    dataset.append(generate_distractor(2, num_distractors=2))  # one extra to hit ~25

    # red herring: ~22 stories, spread across 1-4 transfers
    for num_transfers in [0, 1, 2, 3, 4]:
        for _ in range(5):
            dataset.append(
                generate_red_herring(
                    num_transfers, num_distractors=random.randint(0, 2)
                )
            )
    dataset.append(generate_red_herring(3, num_distractors=1))
    dataset.append(generate_red_herring(1, num_distractors=0))

    dataset.extend(REVERSAL_STORIES)
    dataset.extend(CONTROL_STORIES)

    random.shuffle(dataset)

    for i, item in enumerate(dataset):
        item["id"] = i

    return dataset


if __name__ == "__main__":
    dataset = generate_dataset()

    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} stories")
    print(f"Option 1: {sum(1 for d in dataset if d['type'] == 'distractor')}")
    print(f"Option 2: {sum(1 for d in dataset if d['type'] == 'red_herring')}")
    print(f"Option 3: {sum(1 for d in dataset if d['type'] == 'reversal')}")
