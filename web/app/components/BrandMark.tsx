import styles from "./BrandMark.module.css";

/** Shared brand lockup, previously duplicated in both page components. */
export default function BrandMark() {
  return (
    <span className={styles.brandMark} aria-hidden="true">
      <span className={styles.brandMarkCore} />
      <span className={styles.brandMarkOrbit} />
    </span>
  );
}
