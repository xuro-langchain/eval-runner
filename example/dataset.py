from langsmith import Client

client = Client()

dataset_name = "eli5"

examples = [
    {
        "question": "What is sound?",
        "response": "Okay! Imagine you have a drum. When you hit it, the drum shakes, right? That shaking is called a vibration. \n\nNow, when the drum shakes, it makes the air around it wiggle too, almost like when you throw a rock in a pond and the water makes little waves. These wiggles in the air are what we call sound! \n\nSo, sound is like little invisible wiggles in the air that our ears can feel and hear. Just like how you can hear your friend calling you or your favorite song, those are sounds!"
    },
    {
        "question": "How does string theory work?",
        "response": "Okay! Imagine that everything in the universe, like stars, trees, and even you, is made of tiny building blocks, kind of like LEGO bricks. Now, in string theory, instead of those building blocks being tiny dots, they are super tiny strings, like the strings on a guitar!\n\nThese strings can wiggle and vibrate. Picture a guitar playing different notes – when a string vibrates in different ways, it makes different sounds. In the same way, when these tiny strings vibrate differently, they become different things, like particles that make up everything you see around you.\n\nSo, when the strings wiggle and dance around in space, they interact with each other – sometimes they might hug and combine, and other times they might bounce off each other, just like friends playing together. When we look closely at these strings, they act like the little particles we learn about in science.\n\nIn short, string theory is like saying that everything is made of tiny, wiggly strings, and how they dance and vibrate tells us all about the world we live in!"
    },
    {
        "question": "How does a democracy work?",
        "response": "Okay! Imagine you and your friends want to decide what game to play. Instead of just one person deciding, everyone gets a turn to say what they want. \n\nIn a democracy, it’s like that, but for the whole country! Here’s how it works:\n\n1. **Everyone Has a Voice**: Just like you and your friends all get to say what game you want, in a democracy, everyone gets to share their opinions. This means that everyone gets to vote on important things.\n\n2. **Voting**: When it’s time to make a decision, like picking a leader or a new law, everyone votes. Voting is like marking your favorite game on a piece of paper. The choice with the most votes wins!\n\n3. **Leaders**: The people who get the most votes (like class leaders) are called leaders too! They help make the rules and decisions for everyone.\n\n4. **Fairness**: Everyone is treated equally, so whether you’re big or small, everyone’s voice is important!\n\n5. **Change is Possible**: If people don’t like something, they can vote again later or even choose new leaders. It's like changing the game if everyone wants to play something different.\n\nSo, in a democracy, people work together to decide what’s best for everyone by talking, voting, and making sure everyone can be heard!"
    },
    {
        "question": "Why is the sky blue?",
        "response": "Alright! Imagine the sky is like a big bowl of air, and inside that bowl, there are tiny, tiny bits of dust and gas floating around. When the sun shines, it sends out light that looks white to us, but it has many colors in it, like a rainbow!\n\nNow, different colors of light travel through the air in different ways. Blue light is special because it is shorter and likes to bounce around a lot! So when the sunlight comes into our bowl of air, the blue light gets scattered around in all directions, and that’s what we see when we look up—the blue sky!\n\nIn the evening, when the sun is going down, its light has to travel through more air to reach us. So, the blue light gets scattered even more and can’t reach us as easily. Instead, the other colors, like red and orange, can travel through better, making the sky look like a beautiful sunset with lots of warm colors!\n\nSo, to remember: The sky is blue during the day because blue light is bouncing all around, and it turns red at night because the light has to go a longer way, letting the red colors shine through more!"
    },
    {
        "question": "How does photosynthesis work?",
        "response": "Okay! Imagine plants are like tiny chefs, and they like to make their own food. To do this, they need a few special ingredients: sunlight, air, and water.\n\n1. **Sunlight**: This is like the magic ingredient! Plants catch sunlight with their green leaves. The green color comes from something called chlorophyll, which helps them grab the sun’s light.\n\n2. **Air**: Plants take in a part of the air called carbon dioxide. It’s like a little breath they take in through tiny holes in their leaves.\n\n3. **Water**: Plants drink up water through their roots from the ground, kind of like how we drink water through a straw.\n\nNow, here’s where the magic happens! When the plants have sunlight, air, and water, they mix them all together inside their leaves. This special mixing process creates food for the plant, which is called glucose (that’s just a fancy name for sugar!). \n\nAs a super-duper bonus, when plants make their food, they also release oxygen, which is the air we breathe! So, thanks to photosynthesis, plants not only feed themselves but also help us have the air we need.\n\nSo, in short, photosynthesis is how plants use sunlight, air, and water to make their own food and give us oxygen—what great little chefs they are!"
    },
]

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[{"question": ex["question"]} for ex in examples],
        outputs=[{"response": ex["response"]} for ex in examples],
        dataset_id=dataset.id
    )
    print(f"Dataset '{dataset_name}' created and {len(examples)} examples uploaded.")
else:
    print(f"Dataset '{dataset_name}' already exists.")
