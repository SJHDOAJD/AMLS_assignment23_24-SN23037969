# import code for tasks from folder A and B
from A.TaskA import codeA
from B.TaskB import codeB


def main():

    # start to run Task A code
    print("Running task A...")
    codeA()


    # start to run Task B code
    print("Running task B...")
    codeB()

if __name__ == "__main__":
    main()