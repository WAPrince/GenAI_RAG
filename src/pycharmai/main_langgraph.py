from pycharmai.graph.agent import graph


from pycharmai.graph.agent import run_agent

# Merksatz
# Alles vor interrupt() kann beim Resume erneut ausgeführt werden. Alles nach interrupt() wird erst beim Resume ausgeführt.
# Das ist für deinen Agenten sehr wichtig. Deshalb sollte man vor interrupt() beispielsweise keine irreversiblen Aktionen ausführen:
# run_agent("Lösche bitte die Datei test.txt")


def main() -> None:


    config = {
        "configurable": {
            "thread_id": "conversation-234"
        }
    }

    # Verify if HITL works
    #result = graph.invoke(
    #     {
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": (
    #                     "Delete the file cv1.pdf"
    #                 ),
    #             }
    #         ]
    #     },
    #     config,
    #)

    #result = graph.invoke(
    #    {
    #        "messages": [
    #            {
    #                "role": "user",
    #                "content": (
    #                    "Get the weather for Berlin please"
    #                ),
     #           }
    #        ]
    #    },
    #    config,
    #)


    print(
        result["messages"][-1].content
    )


if __name__ == "__main__":
    main()