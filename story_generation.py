import random
import json
import itertools

random.seed(67)

OBJECTS = ["book", "pen", "notebook", "mug", "key"]
LOCATIONS = ["shelf", "table", "bed", "floor", "desk", "counter"]
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


def generate_distractor(num_transfers, num_distractors):
    obj = random.choice(OBJECTS)
    names = random.sample(NAMES, min(num_transfers + 1, len(NAMES)))
    locations = random.sample(LOCATIONS, min(num_transfers + 1, len(LOCATIONS)))
 
    # build move sentences first
    move_sentences = []
    move_sentences.append(move_sentence(names[0], obj, locations[0]))
    current_location = locations[0]
    old_locations = [locations[0]]
 
    for i in range(1, num_transfers + 1):
        name = names[i] if i < len(names) else random.choice(NAMES)
        loc = (
            locations[i]
            if i < len(locations)
            else random.choice([l for l in LOCATIONS if l != current_location])
        )
        move_sentences.append(move_sentence(name, obj, loc))
        old_locations.append(current_location)
        current_location = loc
 
    # build distractor sentences referencing old locations
    distractors = []
    for _ in range(num_distractors):
        if old_locations:
            old_loc = (
                random.choice(old_locations[:-1])
                if len(old_locations) > 1
                else old_locations[0]
            )
            trap_name = random.choice(NAMES)
            distractors.append(f"{trap_name}'s favorite spot is the {old_loc}.")
 
    # interleave distractors randomly into move sequence
    all_sentences = move_sentences.copy()
    for d in distractors:
        pos = random.randint(0, len(all_sentences))
        all_sentences.insert(pos, d)
 
    all_sentences.append(f"Where is the {obj}?")
 
    return {
        "type": "distractor",
        "num_transfers": num_transfers,
        "num_location_distractors": num_distractors,
        "num_random_distractors": 0,
        "story": " ".join(all_sentences),
        "object": obj,
        "answer": current_location,
    }


def generate_red_herring(num_transfers, num_distractors=1):
    obj = random.choice(OBJECTS)
    other_obj = random.choice([o for o in OBJECTS if o != obj])
 
    names = random.sample(NAMES, min(num_transfers + 2, len(NAMES)))
    locations = random.sample(LOCATIONS, min(num_transfers + 2, len(LOCATIONS)))
 
    # build move sentences for target object
    move_sentences = []
    move_sentences.append(move_sentence(names[0], obj, locations[0]))
    current_location = locations[0]
    old_locations = [locations[0]]
 
    for i in range(1, num_transfers + 1):
        name = names[i] if i < len(names) else random.choice(NAMES)
        loc = (
            locations[i]
            if i < len(locations)
            else random.choice([l for l in LOCATIONS if l != current_location])
        )
        move_sentences.append(move_sentence(name, obj, loc))
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
    red_herring = move_sentence(red_herring_name, other_obj, red_herring_loc)
 
    # random filler distractors
    fillers = [distractor_sentence(random.choice(NAMES)) for _ in range(num_distractors)]
 
    # interleave red herring and fillers randomly into move sequence
    all_sentences = move_sentences.copy()
    for d in [red_herring] + fillers:
        pos = random.randint(0, len(all_sentences))
        all_sentences.insert(pos, d)
 
    all_sentences.append(f"Where is the {obj}?")
 
    return {
        "type": "red_herring",
        "num_transfers": num_transfers,
        "num_location_distractors": 1,
        "num_random_distractors": num_distractors,
        "story": " ".join(all_sentences),
        "object": obj,
        "answer": current_location,
    }

CONTROL_STORIES = []
for obj, loc in itertools.product(OBJECTS, LOCATIONS):
    name = random.choice(NAMES)
    CONTROL_STORIES.append({
        "type": "control",
        "num_transfers": 0,
        "num_location_distractors": 0,
        "num_random_distractors": 0,
        "story": f"{name} puts the {obj} on the {loc}. Where is the {obj}?",
        "object": obj,
        "answer": loc,
    })


def generate_dataset():
    dataset = []

    for num_transfers in [0, 1, 2, 3, 4]:
        for num_distractors in [0, 1, 2]:
            for _ in range(33):
                dataset.append(
                    generate_distractor(num_transfers, num_distractors=num_distractors)
                )

    for num_transfers in [0, 1, 2, 3, 4]:
        for num_distractors in [0, 1, 2]:
            for _ in range(33):
                dataset.append(
                    generate_red_herring(
                        num_transfers, num_distractors=num_distractors
                    )
                )

    random.shuffle(dataset)

    dataset = CONTROL_STORIES + dataset

    for i, item in enumerate(dataset):
        item["id"] = i

    return dataset


if __name__ == "__main__":
    dataset = generate_dataset()

    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} stories")
    print(f"Distractor: {sum(1 for d in dataset if d['type'] == 'distractor')}")
    print(f"Red Herring: {sum(1 for d in dataset if d['type'] == 'red_herring')}")
    print(f"Control: {sum(1 for d in dataset if d['type'] == 'control')}")
