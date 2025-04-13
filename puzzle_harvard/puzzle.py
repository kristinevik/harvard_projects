from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    Or(AKnight, AKnave),  # each character is either a knight or a knave
    Not(And(AKnight, AKnave)),  # can not be both
    # Every sentence spoken by a knight is true,
    # so if A is a knight, it implies that the statement is true - A is both knight and knave
    Biconditional(AKnight, And(AKnight, AKnave)),  
    Biconditional(AKnave, Not(And(AKnight, AKnave)))  # every sentence spoken by a knave is false, so if A is a knave, the statement is false
)
# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    Or(AKnight, AKnave),  # each character is either a knight or a knave
    Or(BKnight, BKnave),  # each character is either a knight or a knave
    Not(And(AKnight, AKnave)),  # can not be both
    Not(And(BKnight, BKnave)),  # can not be both

    # Every sentence spoken by a knight is true. If A is a knight, then the sentence A says "We are both knaves." 
    #  is true - both A and B are knaves
    Biconditional(AKnight, And(AKnave, BKnave)),

    # If A is a knave, it is not true, as "every sentence spoken by a knave is false."
    # Since the statement is not true, and A is a knave - the AI should now logic that B must be a knight.
    Biconditional(AKnave, Not(And(AKnave, BKnave)))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    Or(AKnight, AKnave),  # each character is either a knight or a knave
    Or(BKnight, BKnave),  # each character is either a knight or a knave
    Not(And(AKnight, AKnave)),  # can not be both
    Not(And(BKnight, BKnave)),  # can not be both

    # If A is a knight, then what A is saying is true, so A and B are the same kind
    Biconditional(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    
    # If A is a knave, then A is lying, so A and B are not the same kind
    Biconditional(AKnave, And(Not(And(AKnight, BKnight)), Not(And(AKnave, BKnave)))),

    # If B is a knight, then B is telling the truth, so A and B are different kinds
    Biconditional(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))),

    # If B is a knave, then B is lying, so A and B are not different kinds
    Biconditional(BKnave, And(Not(And(AKnight, BKnave)), Not(And(AKnave, BKnight)))),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    Or(AKnight, AKnave),  
    Or(BKnight, BKnave),  
    Or(CKnight, CKnave),  
    Not(And(AKnight, AKnave)), 
    Not(And(BKnight, BKnave)),  
    Not(And(CKnight, CKnave)), 

    # If A is a knight, then sentence is true
    Biconditional(AKnight, Or(AKnight, AKnave)),

    # If A is a knave, then it is a lie'
    Biconditional(AKnave, Not(Or(AKnight, AKnave))), 

    # If B is a knight, then sentence is true: "A said 'I am a knave'." and "C is a knave."
    # Then dependent on if A is a knight or a knave, what B said that A said is a true statement
    Biconditional(BKnight, And(Or( # Depends if A is kneight or knave
        Biconditional(AKnight, AKnave),  # If A is a knight and A said "I am a knave,", A is telling the truth
        Biconditional(AKnave, Not(AKnave)), # If A is a knave and A said "I am a knave," A is lying
        ), CKnave)),

    # If B is a knave, then sentence about A is false, and we can not infer anything from it
    # If B is a knave, then sentence about C is false
    Biconditional(BKnave, Not(CKnave)),

    # If C is a knight, then sentence is true: C says "A is a knight."
    # If C is a knight, then biconditonally B cannot be a knight
    Biconditional(CKnight, AKnight),

    # If C is a knave, then it is a lie
    Biconditional(CKnave, Not(AKnight)))

def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
