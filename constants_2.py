class Constants:
    #Fitness Function Constants
    DESIRED_FREESPACE_ENTITY_RATIO_PERCENT = 40
    MAX_NUMBER_FREE_ENEMY_CHANGES = 1
    VALUE_REDUCTION_PER_ENEMY = 0.25
    TARGETED_BITMAP_DIFFERENCE = 15

    #Fitness Function Weights : 
    FITNESS_WEIGHT_ENEMIES = 4
    FITNESS_WEIGHT_BALANCE = 3
    FITNESS_WEIGHT_CHANGES = 5
    FITNESS_WEIGHT_SYMMETRY = 5
    FITNESS_WEIGHT_VARIATION = 3
    FITNESS_WEIGHT_POOP_FIRE = 1

    INVERTED_FITNESS_WEIGHT_TOPOLOGY = 1
    INVERTED_FITNESS_WEIGHT_DETOUR = 1
    INVERTED_FITNESS_WEIGHT_EPROX = 3

    #Evolutionary Algorithm Inputs
    NUMBER_GENERATIONS = 50
    CROSSOVER_PROBABILITY = 0.4
    MUTATION_PROBABILITY = 1
    POPULATION_SIZE = 100
    NUMBER_ELITES = 10

    #Tile probability 
    PROB_FREE_SPACE = 20
    PROB_FIRE = 4
    PROB_STONE = 15
    PROB_POOP = 4
    PROB_ENTITY = 6
    PROB_BLOCK = 0
    PROB_PIT = 0
    PROB_SPIKE = 0

    


    # Änderungen tboi_ea
    # Anpassung der Tile-Wahrscheinlichkeiten und löschen von Block und Spike und Pit
    # Während Generation der Bitmaps ändere Targeted Bitmap Difference immer ein wenig um mehr unterschiedliche Bitmaps zu generieren (Von 30 - 55)
    # Während Generation der Bitmaps ändere DESIRED_FREESPACE_ENTITY_RATIO_PERCENT immer ein wenig um mehr unterschiedliche Bitmaps zu generieren (Von 30 - 50)
    # Füge der Fitness Funktion einen Score hinzu, welcher die Funktion positiv beeinflusst, wenn Poop und Fire erreichbar sind
    # Variation leicht nach unten um Fire und Poop nicht zu hart zu bestrafen
