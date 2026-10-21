# Third-party notices

Mumble Rhyme's original code is licensed under GPL-3.0-or-later; see [LICENSE](LICENSE). Dependencies, dictionary data, and optional model assets retain their own licenses. This repository and its built packages do not bundle model weights or the dictionary distribution.

## Pronunciation wrapper

The installed dependency `cmudict==1.1.3` is the Python wrapper maintained by David L. Day and contributors. Its declared license is **GPL-3.0-or-later**. Project: <https://github.com/prosegrinder/python-cmudict>. Its license text is supplied with that dependency. The Mumble Rhyme code uses GPL-3.0-or-later to match the current wrapper dependency.

## Dictionary data

The CMU Pronouncing Dictionary data has separate Carnegie Mellon redistribution terms. Upstream: <https://github.com/cmusphinx/cmudict>. The following is the data license from the installed distribution:

```text
Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   The contents of this file are deemed to be source code.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.

This work was supported in part by funding from the Defense Advanced
Research Projects Agency, the Office of Naval Research and the National
Science Foundation of the United States of America, and by member
companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
the contributions of many volunteers to the expansion and improvement of
this dictionary.

THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Optional model and tokenizer

`Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, is obtained separately from Hugging Face. Its upstream license is **Apache-2.0**. The downloaded snapshot contains its LICENSE. Local NF4 conversion and Windows file resharding do not transfer ownership or replace those terms.

Model card: <https://huggingface.co/Qwen/Qwen3.5-9B>. Fixed snapshot: <https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a>.

The optional `Qwen/Qwen3.5-4B` selection is also downloaded separately under Apache-2.0, at fixed revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Its matching tokenizer comes from the same snapshot: <https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a>.

## Runtime dependencies

Other directly used projects include PyYAML, Pydantic, the MCP Python SDK, PyTorch, Transformers, Accelerate, bitsandbytes, Hugging Face Hub, and their transitive dependencies. They are installed separately; their distributions supply their license notices. `requirements-lock.txt` is a record of the verified Windows environment, not a relicensing of those packages.
