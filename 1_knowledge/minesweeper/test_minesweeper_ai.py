from minesweeper import MinesweeperAI

def test_infer_mine():
    ai = MinesweeperAI(height=3, width=3)
    ai.add_knowledge((1, 1), 1)
    ai.add_knowledge((1, 2), 1)
    ai.add_knowledge((2, 1), 1)
    assert ai.mines == {(2, 2)}, f"Expected {(2, 2)}, but got {ai.mines}"

def test_infer_multiple_mines():
    ai = MinesweeperAI(height=3, width=3)
    ai.add_knowledge((0, 0), 2)
    ai.add_knowledge((0, 1), 2)
    assert ai.mines == {(1, 0), (1, 1)}, f"Expected {(1, 0), (1, 1)}, but got {ai.mines}"

def test_infer_safe_cells():
    ai = MinesweeperAI(height=3, width=3)
    ai.add_knowledge((1, 1), 1)
    ai.add_knowledge((1, 2), 1)
    assert (0, 0) in ai.safes, f"Expected (0, 0) to be in safes, but it was not found"

if __name__ == "__main__":
    test_infer_mine()
    test_infer_multiple_mines()
    test_infer_safe_cells()
    print("All tests passed!")
