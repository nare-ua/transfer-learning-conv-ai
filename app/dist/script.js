class Api {
    path(p) {
        return `${Api.ENDPOINT}/${p}`;
    }
    async getShuffle() {
        const response = await fetch('/shuffle');
        return response.json();
    }
    async postMessage(user, query, params) {
        const qs = new URLSearchParams(query);
        const path = this.path(`messages/${user}?${qs.toString()}`);
        const response = await fetch(path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        return await response.json();
    }
    /**
     * Demo-specific helpers
     */
    async postWithSettings(query, params) {
        /// Retrieve all settings params then launch the request.
        const top_k = Number(document.querySelector('.decoder-settings .setting:nth-child(2) .js-val').textContent);
        const top_p = Number(document.querySelector('.decoder-settings .setting:nth-child(3) .js-val').textContent);
        const temperature = Number(document.querySelector('.decoder-settings .setting:nth-child(4) .js-val').textContent);
        return this.postMessage(`toto`, query, Object.assign({}, params, { top_k, top_p, temperature }));
    }
}
Api.ENDPOINT = `http://221.147.119.95:5000`
Api.shared = new Api();
class Markup {
    static scrollToBottom(opts = {}) {
        App.messagesRoot.scrollTop = App.messagesRoot.scrollHeight;
    }
    static messageMarkup(m) {
        const incomingStr = m.incoming ? 'incoming' : 'outgoing';
        return `<div class="message ${incomingStr}">
			<div class="message-inner">${Utils.escape(m.content)}</div>
		</div>`;
    }
    static append(m) {
        const s = this.messageMarkup(m);
        App.messagesRoot.insertAdjacentHTML('beforeend', s);
        this.scrollToBottom();
    }
    /**
     * Bucketize a float into a level
     * according to a set of thresholds.
     */
    static attentionThreshold(att) {
        const thresholds = [2, 4.5, 10, 30];
        for (const [i, x] of thresholds.entries()) {
            if (x > att) {
                return i;
            }
        }
        return thresholds.length;
    }
}
const App = {
    persona: {
        slug: "",
        text: "",
    },
    messages: [],
    /// HTMLElements
    messagesRoot: document.querySelector('div.messages'),
    divPersona: document.querySelector('div.persona'),
    linkSuggest: document.querySelector('.chat-suggestion .js-suggestion'),
    sliders: Array.from(document.querySelectorAll('.decoder-settings input.slider')),
};
document.addEventListener('DOMContentLoaded', () => {
    /**
     * Persona
     * - in simple mode, we just re-format the text slightly to make it prettier.
     * - in attention mode, we tokenize and add spans around every token.
     */
    const simplePersona = () => {
        const text = App.persona.text;
        const html = text.split('.').map(x => Utils.capitalize(x)).join(`.<br>`);
        App.divPersona.innerHTML = html;
    };
    const tokenizePersona = () => {
        const text = App.persona.text;
        const tokens = text.split(/\b/).filter(x => !/\s/.test(x));
        let html = ``;
        for (let [i, tok] of tokens.entries()) {
            if (i === 0) {
                html += `<span data-idx="${i}">${Utils.capitalize(tok)}</span>`;
            }
            else if (i > 0 && tokens[i - 1] === '.') {
                html += `<br><span data-idx="${i}">${Utils.capitalize(tok)}</span>`;
            }
            else if (/[.,']/.test(tok)) {
                html += `<span data-idx="${i}">${tok}</span>`;
            }
            else {
                html += ` <span data-idx="${i}">${tok}</span>`;
            }
        }
        App.divPersona.innerHTML = html;
    };
    App.persona = window.PERSONA_ATLOAD;
    // ^^ tokenizePersona();
    simplePersona();
    document.querySelector('.js-shuffle').addEventListener('click', async (evt) => {
        evt.preventDefault();
        App.persona = await Api.shared.getShuffle();
        tokenizePersona();
        history.replaceState(null, "", `/persona/${App.persona.slug}`);
        /// Also reset messages and reload suggestion.
        App.messages = [];
        App.messagesRoot.innerHTML = "";
        loadSuggestion();
    });
  /**
    * document.querySelector('.js-share').addEventListener('click', async (evt) => {
    *     evt.preventDefault();
    *     history.replaceState(null, "", `/persona/${App.persona.slug}`);
    *     const text = `Chat with me: ${App.divPersona.innerText.replace(/\n/g, " ")}`;
    *     window.open(`https://twitter.com/share?url=${encodeURIComponent(window.location.href)}&text=${encodeURIComponent(text)}`);
    * });
    */

    /**
     * Settings
     */
    const handleSliderChange = (slider) => {
        const div = slider.parentNode;
        const spanVal = div.querySelector('.js-val');
        const value = Number.isInteger(slider.valueAsNumber)
            ? slider.valueAsNumber
            : Number(slider.valueAsNumber.toFixed(2));
        spanVal.innerText = value.toString();
        const min = Number(slider.getAttribute('min'));
        const max = Number(slider.getAttribute('max'));
        if (value < min + (max - min) / 3) {
            spanVal.className = "js-val green";
        }
        else if (value < min + 2 * (max - min) / 3) {
            spanVal.className = "js-val orange";
        }
        else {
            spanVal.className = "js-val red";
        }
        const isInverted = slider.classList.contains('js-inverted');
        if (isInverted) {
            if (spanVal.classList.contains('green')) {
                spanVal.classList.remove('green');
                spanVal.classList.add('red');
            }
            else if (spanVal.classList.contains('red')) {
                spanVal.classList.remove('red');
                spanVal.classList.add('green');
            }
        }
    };
    for (const slider of App.sliders) {
        handleSliderChange(slider);
        slider.addEventListener('input', () => {
            handleSliderChange(slider);
        });
    }
    const gauge = document.querySelector('div.gauge');
    gauge.addEventListener('click', () => {
        gauge.classList.add('active');
    });
    const gaugeEls = Array.from(document.querySelectorAll('.gauge .gauge-el-wrapper'));
    for (const gaugeEl of gaugeEls) {
        gaugeEl.addEventListener('click', () => {
            const i = gaugeEls.indexOf(gaugeEl);
            if (i === 0) {
                App.sliders[0].value = `180`;
                App.sliders[1].value = `0.1`;
                App.sliders[2].value = `1.9`;
            }
            else if (i === 1) {
                App.sliders[0].value = `70`;
                App.sliders[1].value = `0.5`;
                App.sliders[2].value = `1.2`;
            }
            else {
                App.sliders[0].value = `0`;
                App.sliders[1].value = `0.9`;
                App.sliders[2].value = `0.6`;
            }
            for (const slider of App.sliders) {
                handleSliderChange(slider);
            }
        });
    }
    /**
     * Chat input
     */
    const form = document.querySelector('form.js-form');
    const input = document.querySelector('input.input-message');
    form.addEventListener('submit', async (evt) => {
        evt.preventDefault();
        document.querySelector('.placeholder-start').style.display = 'none';
        const content = input.value;
        if (content.trim() === "") {
            c.debug(`Empty input`);
            return;
        }
        input.value = "";
        const um = {
            incoming: false,
            content: content,
        };
        Markup.append(um);
        const o = await Api.shared.postWithSettings({ text: content }, {
            context: App.messages,
            persona: App.persona.text,
        });
        c.log(o);
        // c.log(o.attention[0]);
        // c.log(o.attention[1]);
        // c.log(o.attention[2]);
        App.messages.push(um);
        /// ^^ not before the API call because it shouldn't be included in context.
        /**
         * Visualize Attention
         */
        // for (const [idx, [token, att]] of o.attention[0].entries()) {
        // 	if (att === 0) {
        // 		continue;
        // 	}
        // 	const span = document.querySelector(`.persona span[data-idx="${idx}"]`);
        // 	if (!span) {
        // 		c.error(`span not found`);
        // 		return ;
        // 	}
        // 	span.className = `attention-level level${ Markup.attentionThreshold(att) }`;
        // }
        //////
        const m = {
            incoming: true,
            content: o.text,
        };
        App.messages.push(m);
        Markup.append(m);
        /// Finally, launch an auto-complete:
        loadSuggestion();
    });
    /**
     * Suggestion box
     */
    const loadSuggestion = async () => {
      const spanLoading = document.querySelector('.chat-suggestion .js-loading');
      spanLoading.classList.remove('hide');
      App.linkSuggest.classList.add('hide');
      const o = await Api.shared.postWithSettings({ completion: "" }, {
          context: App.messages,
          persona: App.persona.text,
      });
      c.log(o);
      App.linkSuggest.innerText = o.text;
      spanLoading.classList.add('hide');
      App.linkSuggest.classList.remove('hide');
    };

    loadSuggestion();
    App.linkSuggest.addEventListener('click', async (evt) => {
        evt.preventDefault();
        App.linkSuggest.classList.add('hide');
        input.value = App.linkSuggest.innerText;
        await Utils.delay(500);
        const sendBtn = form.querySelector('button.input-button');
        sendBtn.click();
        /// ^^ do not do `form.submit()` as we want to trigger our handler.
    });
});
const c = console;
class Utils {
    /**
     * Escape a message's content for insertion into html.
     */
    static escape(s) {
        let x = s;
        for (const [k, v] of Object.entries(this.escapeMap)) {
            x = x.replace(new RegExp(k, 'g'), v);
        }
        return x.replace(/\n/g, '<br>');
    }
    /**
     * Opposite of escape.
     */
    static unescape(s) {
        let x = s.replace(/<br>/g, '\n');
        for (const [k, v] of Object.entries(this.escapeMap)) {
            x = x.replace(new RegExp(v, 'g'), k);
        }
        return x;
    }
    /**
     * "Real" modulo (always >= 0), not remainder.
     */
    static mod(a, n) {
        return ((a % n) + n) % n;
    }
    /**
     * Noop object with arbitrary number of nested attributes that are also noop.
     */
    static deepNoop() {
        const noop = new Proxy(() => { }, {
            get: () => noop
        });
        return noop;
    }
    /**
     * Capitalize
     */
    static capitalize(s) {
        return s.charAt(0).toUpperCase() + s.slice(1);
    }
    /**
     * Returns a promise that will resolve after the specified time
     * @param ms Number of ms to wait
     */
    static delay(ms) {
        return new Promise((resolve, reject) => {
            setTimeout(() => resolve(), ms);
        });
    }
}
Utils.escapeMap = {
    /// From underscore.js
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
    '`': '&#x60;'
};
