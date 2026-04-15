/**
 * WindowManager — Drag, Resize, Dock/Undock for JsBrain panels
 */
export class WindowManager {
    constructor() {
        this.panels = new Map();
        this._topZ = 500;
        this._dragState = null;
        this._resizeState = null;

        document.addEventListener('mousemove', (e) => this._onMouseMove(e));
        document.addEventListener('mouseup', () => this._onMouseUp());
    }

    register(panelId, options = {}) {
        const el = document.getElementById(panelId);
        if (!el) return;

        const dragHandle = el.querySelector('.panel-header');

        this._injectWMButtons(el, panelId);

        const resizeHandle = document.createElement('div');
        resizeHandle.className = 'resize-handle';
        el.appendChild(resizeHandle);

        this.panels.set(panelId, {
            element: el,
            placeholder: null,
            state: 'docked',
            dragHandle,
            resizeHandle,
            originalParent: el.parentElement,
            originalNextSibling: el.nextElementSibling,
            // Capture ALL original computed layout attributes
            originalCssText: el.style.cssText,
            rect: { x: 100, y: 100, w: 500, h: 400 },
            minW: options.minW || 300,
            minH: options.minH || 200,
        });

        if (dragHandle) {
            dragHandle.addEventListener('mousedown', (e) => {
                const panel = this.panels.get(panelId);
                if (panel.state !== 'floating') return;
                if (e.target.closest('.wm-btn') || e.target.closest('button')) return;
                e.preventDefault();
                this._dragState = {
                    panelId,
                    startX: e.clientX,
                    startY: e.clientY,
                    startPanelX: panel.rect.x,
                    startPanelY: panel.rect.y
                };
            });
        }

        resizeHandle.addEventListener('mousedown', (e) => {
            const panel = this.panels.get(panelId);
            if (panel.state !== 'floating') return;
            e.preventDefault();
            e.stopPropagation();
            this._resizeState = {
                panelId,
                startX: e.clientX,
                startY: e.clientY,
                startW: panel.rect.w,
                startH: panel.rect.h
            };
        });

        el.addEventListener('mousedown', () => {
            const panel = this.panels.get(panelId);
            if (panel.state === 'floating') this._bringToFront(panelId);
        });
    }

    _injectWMButtons(el, panelId) {
        const header = el.querySelector('.panel-header');
        if (!header) return;

        const btnContainer = document.createElement('div');
        btnContainer.className = 'panel-actions wm-actions';
        btnContainer.style.cssText = 'display:flex;gap:4px;margin-left:auto;';

        const btnUndock = document.createElement('button');
        btnUndock.className = 'icon-btn wm-btn';
        btnUndock.title = 'Undock / Dock';
        btnUndock.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/><path d="M9 9h12"/></svg>`;
        btnUndock.addEventListener('click', () => {
            const panel = this.panels.get(panelId);
            if (panel.state === 'docked') this.undock(panelId);
            else this.dock(panelId);
        });

        btnContainer.appendChild(btnUndock);

        const existing = header.querySelector('.panel-actions:not(.wm-actions)');
        if (existing) existing.after(btnContainer);
        else header.appendChild(btnContainer);
    }

    undock(panelId) {
        const panel = this.panels.get(panelId);
        if (!panel || panel.state === 'floating') return;

        const el = panel.element;
        const rect = el.getBoundingClientRect();

        // Save the full original inline style
        panel.originalCssText = el.style.cssText;

        // Create placeholder to keep layout intact
        const placeholder = document.createElement('div');
        placeholder.className = 'panel-placeholder';
        placeholder.dataset.panelId = panelId;
        // Copy the original flex/size styles so layout doesn't shift
        placeholder.style.cssText = panel.originalCssText;
        el.parentElement.insertBefore(placeholder, el);
        panel.placeholder = placeholder;

        panel.rect = {
            x: rect.left + 20,
            y: rect.top + 20,
            w: Math.max(rect.width, panel.minW),
            h: Math.max(rect.height, panel.minH),
        };

        document.body.appendChild(el);
        el.classList.add('floating');
        el.classList.remove('maximized');

        this._applyRect(panelId);
        this._bringToFront(panelId);

        panel.resizeHandle.style.display = 'block';
        panel.state = 'floating';

        setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
    }

    dock(panelId) {
        const panel = this.panels.get(panelId);
        if (!panel || panel.state === 'docked') return;

        const el = panel.element;

        // Return element to its original DOM position
        if (panel.placeholder && panel.placeholder.parentElement) {
            panel.placeholder.parentElement.insertBefore(el, panel.placeholder);
            panel.placeholder.remove();
            panel.placeholder = null;
        } else if (panel.originalParent) {
            if (panel.originalNextSibling && panel.originalNextSibling.parentElement === panel.originalParent) {
                panel.originalParent.insertBefore(el, panel.originalNextSibling);
            } else {
                panel.originalParent.appendChild(el);
            }
        }

        // Restore ALL original inline styles exactly
        el.classList.remove('floating', 'maximized');
        el.style.cssText = panel.originalCssText;

        panel.resizeHandle.style.display = 'none';
        panel.state = 'docked';

        setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
    }

    _bringToFront(panelId) {
        this._topZ++;
        const panel = this.panels.get(panelId);
        if (panel) panel.element.style.zIndex = this._topZ;
    }

    _applyRect(panelId) {
        const panel = this.panels.get(panelId);
        if (!panel) return;
        const el = panel.element;
        el.style.position = 'fixed';
        el.style.left = panel.rect.x + 'px';
        el.style.top = panel.rect.y + 'px';
        el.style.width = panel.rect.w + 'px';
        el.style.height = panel.rect.h + 'px';
        el.style.flex = 'none';
    }

    _onMouseMove(e) {
        if (this._dragState) {
            const ds = this._dragState;
            const panel = this.panels.get(ds.panelId);
            if (!panel) return;
            panel.rect.x = Math.max(0, Math.min(ds.startPanelX + (e.clientX - ds.startX), window.innerWidth - 100));
            panel.rect.y = Math.max(0, Math.min(ds.startPanelY + (e.clientY - ds.startY), window.innerHeight - 50));
            this._applyRect(ds.panelId);
        }
        if (this._resizeState) {
            const rs = this._resizeState;
            const panel = this.panels.get(rs.panelId);
            if (!panel) return;
            panel.rect.w = Math.max(panel.minW, rs.startW + (e.clientX - rs.startX));
            panel.rect.h = Math.max(panel.minH, rs.startH + (e.clientY - rs.startY));
            this._applyRect(rs.panelId);
        }
    }

    _onMouseUp() {
        if (this._dragState || this._resizeState) {
            this._dragState = null;
            this._resizeState = null;
            window.dispatchEvent(new Event('resize'));
        }
    }
}
