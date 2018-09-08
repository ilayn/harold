# How to contribute to harold?

A big thank you in advance, in case you are considering to contribute to `harold`. Contributions, regardless of their size and content, are always welcome. Below you can find some pointers to make things a bit more easier for your workflow.

### Opening an issue

If you have a specific question about the usage of `harold` or you would like to point out a certain detail you can always open an issue. You can also use issues for requesting features that you wish to see. This also gives a chance for the developers to provide their own reasoning in case something doesn't seem possible or not desired.

In case you have a technical problem and you want to solve it using `harold`, you might or might not get a satisfactory answer. Unfortunately it is up to the will of the others whether to work on your problem. Hence it is kindly requested to not offload a control problem in its entirety as an issue. Use Gitter chat room for such things. The link is on the GitHub repository description.

The developers are experts on making mistakes. Thus, please take the time to drop off a comment with the offending input to demonstrate our mistakes. It would be great if we can copy paste and run on our own local machines to reduce the confusion.

### Contributing and sending PRs

Opening a pull request is always thrilling. Thanks to the magical world of Git syntax, it feels a bit like defusing a bomb under pressure. But rest assured, you cannot harm others' work from your own clone (famous last words). Hence feel free to experiment on your local clone ( [Obligatory XKCD reference](https://xkcd.com/1597/) ). There are multiple ways to work with repositories but here is a relatively easy one:

  1. Install Git. It doesn't matter if you use a GUI version or command line or other versions. But command line seems to have more support and flexibility.
  2. Fork the repository on GitHub from the top right corner to your own GitHub. This will be your own copy of `harold` repository that you can modify.
  
  2. By finding the colored (currently green) **Clone or Download** button (top right corner of the file list) to get the cloning address of the repository. And using that address, clone the repository under your favorite folder. You can do this via the following command which will create a folder named `harold` and put the repository inside it.
  
        ```
        git clone https://github.com/ <yourname> /harold.git
        ```
  3. By default you will be on your master git branch. To work on changes, it is always better to keep relevant changes inside their own branches. By doing so, you can keep working on other things on other branches while keeping changes exclusive. Suppose we want to create a branch called `fix_some_bug`. That's done via

        ```
        git checkout -b fix_some_bug
        ```

  4. Now you can implement your changes to the code. Once you are done or want to save things to continue later, you can commit your changes to that branch. You can commit as many times as you want

        ```
        git commit -a -m "Fixed the bode plot bug"
        ```

  5. Almost there! Once you are done with your changes, you have to push your modified code to GitHub. Remember, every change should be committed by now in the previous step. Then type in

        ```
        git push origin
        ```

You can then go to your GitHub repository and switch your branch to the one you were working on and push the button **Create Pull Request**. Then include the details of what you have done and you are done. Awesome!

# Style and related preferences

### Coding style

We follow PEP8 standards but we really don't know why. Hence we don't have a reason not to. That's why. Wonderful recursive logic.

### Issue/PR indicators and abbreviations

For identifying what the PR or the issue is about, you can place a three letter abbreviation to your issue/PR titles. We closely follow the [NumPy style](http://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html). Below, the relevant ones are replicated for convenience

```
BLD: change related to building harold
BUG: bug fix
DEP: deprecate something, or remove a deprecated object
DEV: development tool or utility
DOC: documentation
ENH: enhancement
MAINT: maintenance commit (refactoring, typos etc.)
REV: revert an earlier commit
STY: style fix (whitespace, PEP8)
TST: addition or modification of tests
```

For example, if you have changed only the documentation of some function `func`, your PR title can be `DOC: Removed @ilayn's email password from the docs` which unfortunately can be a real PR :)

### Linking PRs to issueson GitHub

Suppose you have contributed a PR to `harold` about fixing a previously opened issue (again Awesome!).  A nice property of GitHub is that you can link these two by placing any of `Closes` or `Fixes` followed by a hashtag `#` and typing the issue number in the PR description. For example if you add `Closes #89` then when the PR is merged the related issue is also automatically closed. Probably someone else will fix this if you forget it but it is good to keep in mind.

