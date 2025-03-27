from gtts import gTTS
import os

def create_test_speech():
    try:
        text = """
        Welcome to this test recording.
        We are creating a short podcast segment.
        This will help us test the video generation system.
        Each sentence should become its own segment.
        The segmentation algorithm will process this naturally.
        Let's see how the enhancement process works.
        We should see some interesting visual interpretations.
        This is nearing the end of our test.
        Thank you for listening to this test.
        """
        
        text = """
        In Ancient Egypt, the construction of the Great Pyramids of Giza demonstrated remarkable engineering abilities around 2500 BCE. Vibrant hieroglyphic writing covered the walls of the Temple of Karnak while pharaohs like Ramses II ruled as god-kings over the fertile Nile valley. The Book of the Dead guided souls through the afterlife, and alabaster canopic jars preserved organs of the mummified dead within elaborate sarcophagi.
        Moving to Ancient Greece, Athens developed the first democratic system of government in the Agora around 500 BCE. The Parthenon, with its perfect Doric columns, dominated the Acropolis skyline. Philosophers like Socrates, Plato, and Aristotle established methods of logical inquiry at the Lyceum that would influence thought for millennia, while Greek mythology featuring Zeus and the Olympian gods permeated their culture. Red-figure pottery depicted heroic tales of Achilles and Odysseus from Homer's epics.
        The Roman Empire expanded significantly under Julius Caesar's leadership during the 1st century BCE. The Colosseum hosted gladiatorial combat while citizens relaxed in the thermae of the Baths of Caracalla. Later, Emperor Constantine would convert to Christianity at the Milvian Bridge, forever changing the Empire's religious landscape. Roman concrete in the Pantheon's perfect dome, aqueducts like Pont du Gard, and the Appian Way road system demonstrated their engineering prowess throughout the Mediterranean.
        During the Medieval period, feudal systems dominated Europe, with lords in stone keeps, knights in chainmail armor, and peasants in wattle-and-daub cottages forming a strict social hierarchy. Gothic cathedrals like Notre-Dame de Paris with their flying buttresses and stained glass rose windows reached toward the heavens while Benedictine monasteries preserved ancient texts through illuminated manuscripts. The Black Death devastated populations in the 14th century, with victims displaying the characteristic bubonic symptoms.
        The Renaissance period saw a rebirth of classical learning beginning in 15th century Florence. Artists like Leonardo da Vinci painted "The Last Supper" in tempera and Michelangelo sculpted "David" from Carrara marble while Gutenberg's movable-type printing press revolutionized information sharing in Mainz around 1450. Lorenzo de' Medici patronized the arts from his Palazzo Medici, and Brunelleschi's octagonal dome crowned the Florence Cathedral using innovative herringbone brickwork.
        In the Industrial Revolution of the 18th and 19th centuries, Watt's steam engine transformed manufacturing at factories like Cromford Mill, while transportation evolved with Stephenson's "Rocket" locomotive on the Liverpool-Manchester Railway. Textile production moved from spinning wheels to Arkwright's water frame and Cartwright's power loom. Manchester's cotton mills and Birmingham's iron foundries belched coal smoke as rural populations migrated to terraced housing in rapidly growing industrial centers.
        The modern era brought rapid technological innovation, with Ford's Model T automobiles, the Wright brothers' flights at Kitty Hawk, IBM mainframe computers, and ARPANET fundamentally changing human society. During the Cold War, tensions between nuclear superpowers threatened global security from the Berlin Wall to the Cuban Missile Crisis, while the information age connected people worldwide through fiber optic cables, silicon microchips, and TCP/IP protocols.
        Today in the contemporary world, neural network artificial intelligence, lithium-ion battery storage for renewable energy, and CRISPR gene editing biotechnology promise new frontiers while humanity faces challenges from rising atmospheric carbon dioxide and evolving social structures. SpaceX's reusable Falcon rockets and the International Space Station continue to expand our cosmic understanding beyond Earth's boundaries, while everyday life is increasingly influenced by smartphones, cloud computing, and social media platforms.
        """

        print("Generating speech from text...")
        tts = gTTS(text=text, lang='en', slow=False)
        
        print("Saving to short_test.wav...")
        tts.save("short_test.wav")
        
        print("Test audio file created successfully!")
        
    except Exception as e:
        print(f"Error creating test speech: {str(e)}")
        print("Note: This requires an internet connection as it uses Google's TTS service")

if __name__ == '__main__':
    create_test_speech()