## Description and Motivation

<!--

    Please write a description of what this PR is changing, removing or adding, and why.
    Consider including before/after comparisons.

    For this kit, a good description usually covers:
      * which knob, default, script, or profile-planner path changes
      * whether the change affects measured numbers (VRAM, context, load, tok/s)
        and, if so, the expected direction
      * why the change is safe on both Windows and Linux launchers
      * for measurement claims: the exact protocol (GPU, budget, quant, context)

-->

## Related Issues

<!--

    Add the list of issues related to this PR from the issue tracker.
    Indicate which of these issues are resolved or fixed by this PR, like #XXXX, where XXXX is the issue number.

-->

---

## Testing

<!--

    Tell us how you verified this change. For this kit that usually means:

      * `bash -n linux/*.sh` (syntax check) for Linux script changes
      * an actual setup/start on Windows and/or Linux with the change
      * if behavior changed, the measured numbers with the new settings
        in the README's format (VRAM budget, context, images on/off)

-->

---

## Checklist:

<!--

    Thanks for contributing to Mia's AI Lab!

    Before you file this pull request, please follow the items on this checklist and
    put an x in each of the boxes, like this: [x].

-->

- [ ] I have read the README and kept my changes consistent with it.
- [ ] My pull request has a sound title and description (not something vague like `Update README.md`).
- [ ] My change is reproducible and verified (a setup/start and, for behavior changes, measurements).
- [ ] Defaults still work out of the box; a new knob has a sane fallback consistent with the existing ones.
- [ ] I updated the README for any knob, default, or measured number I changed.
