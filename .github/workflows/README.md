The contents of this directory are convoluted, as GitHub requires that all workflows reside in this directory (nested directories are not allowed). Therefore, the following naming convention is used for files in the directory:

`callable_`: Callable workflows that are invoked by other workflows in this repository.

`event_`: Workflows that are triggered by an event (pr, push, periodic, etc.).

`manual_`: Workflows that can be triggered by a human on GitHub. These workflows should invoke functionality similar to workflows invoked by events.

`sca_`: Static Code Analysis (SCA) workflows triggered by PRs, commits to mainline branches, and periodic invocations.
