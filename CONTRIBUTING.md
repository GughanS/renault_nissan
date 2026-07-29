# Contributing to WheelEye

First off, thank you for considering contributing to WheelEye! It's people like you that make WheelEye such a great tool for industrial manufacturing automation.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check our [Issues](../../issues) to see if someone else in the community has already created a ticket. If not, go ahead and [make one](../../issues/new).

## Fork & create a branch

If this is something you think you can fix, then fork WheelEye and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```sh
git checkout -b 325-add-tire-pressure-inference
```

## Implementation Guidelines

- **Keep it light**: The core philosophy of WheelEye is speed. Do not introduce heavy dependencies unless absolutely necessary.
- **Test your code**: Ensure you run `pytest` before submitting a PR.
- **Format your code**: We follow standard PEP-8.

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface.
3. You may merge the Pull Request in once you have the sign-off of two other developers.
