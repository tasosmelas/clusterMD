"""Native process transport; scientific flags and selections belong to adapters."""
import subprocess
import shlex
from pathlib import Path
from clustermd.models import PipelineExecutionError

def run_native(command, *, stdin=None, cwd=None, log_path=None):
    command = [str(value) for value in command]
    def log(message):
        if log_path is not None:
            with Path(log_path).open('a') as handle:
                handle.write(message + '\n')
    log('$ ' + shlex.join(command))
    if stdin is not None:
        log('stdin: ' + repr(stdin))
    try:
        result = subprocess.run(command, input=stdin, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=cwd, check=False)
    except FileNotFoundError as exc:
        message = f"Native executable {command[0]!r} was not found."
        log(message)
        raise PipelineExecutionError(message) from exc
    log(result.stdout)
    if result.returncode:
        raise PipelineExecutionError(f"Native command failed (exit {result.returncode}):\n"
                                     f"{shlex.join(command)}\n\n{result.stdout}")
    return result
