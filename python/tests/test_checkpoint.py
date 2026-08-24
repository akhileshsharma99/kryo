import os
import signal
import unittest
from unittest.mock import ANY, call, patch

import kryo


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        kryo._checkpointed = False

    def test_signals_cli_pid_and_waits_for_restore(self) -> None:
        with (
            patch.dict(os.environ, {"KRYO_CLI_PID": "1234"}),
            patch("os.kill") as kill,
            patch("signal.signal") as install_handler,
            patch("time.sleep") as sleep,
        ):

            def deliver_wake(_seconds: float) -> None:
                wake_handler = install_handler.call_args_list[0].args[1]
                wake_handler(signal.SIGUSR2, None)

            sleep.side_effect = deliver_wake
            kryo.checkpoint()

        kill.assert_called_once_with(1234, signal.SIGUSR1)
        sleep.assert_called_once_with(0.1)
        self.assertEqual(
            install_handler.call_args_list,
            [
                call(signal.SIGUSR2, ANY),
                call(signal.SIGUSR2, install_handler.return_value),
            ],
        )

    def test_falls_back_to_parent_for_older_cli(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.getppid", return_value=4321),
            patch("os.kill") as kill,
            patch("signal.signal") as install_handler,
            patch("time.sleep") as sleep,
        ):

            def deliver_wake(_seconds: float) -> None:
                wake_handler = install_handler.call_args_list[0].args[1]
                wake_handler(signal.SIGUSR2, None)

            sleep.side_effect = deliver_wake
            kryo.checkpoint()

        kill.assert_called_once_with(4321, signal.SIGUSR1)


if __name__ == "__main__":
    unittest.main()
